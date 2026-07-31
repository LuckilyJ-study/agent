from __future__ import annotations

from dataclasses import dataclass

from .action_executors import (
    DryRunPolicyBackend,
    DryRunRobotController,
    PolicyExecutor,
    PolicyRegistry,
    PrimitiveRobotController,
    RobotPrimitiveExecutor,
)
from .capabilities import CapabilityRegistry
from .closed_loop import (
    ActionVerifier,
    ClosedLoopAgent,
    ClosedLoopPlanner,
    PlaceholderActionVerifier,
    PreActionGate,
    RecoveryPolicy,
)
from .perception import AgentStateProvider, NullPerceptionProvider, PerceptionProvider
from .policy_metadata import PolicyMetadata
from .persistence import TaskStore
from .control import ExecutionControl
from .domain import WorldState
from .local_recovery import ControllerLocalRecoveryHandler, LocalRecoveryHandler
from .monitor import StructuredActionMonitor
from .safety_monitor import RuntimeSafetyMonitor, SoftwareSafetyMonitor
from .task_verifier import TaskVerifier
from .routing import ExecutorRouter
from .physical_target_gate import (
    PhysicalTargetGate,
    validate_physical_perception_provider,
)


@dataclass
class AgentRuntime:
    agent: ClosedLoopAgent
    state_provider: AgentStateProvider
    capabilities: CapabilityRegistry
    policies: PolicyRegistry
    router: ExecutorRouter


def build_agent_runtime(
    planner: ClosedLoopPlanner,
    *,
    controller: PrimitiveRobotController | None = None,
    perception: PerceptionProvider | None = None,
    policies: PolicyRegistry | None = None,
    verifier: ActionVerifier | None = None,
    recovery_policy: RecoveryPolicy | None = None,
    task_store: TaskStore | None = None,
    safety_monitor: RuntimeSafetyMonitor | None = None,
    task_verifier: TaskVerifier | None = None,
    control: ExecutionControl | None = None,
    initial_world_state: dict | None = None,
    capabilities: CapabilityRegistry | None = None,
    dry_run: bool = True,
    max_replans: int = 3,
    local_recovery: LocalRecoveryHandler | None = None,
    hardware_mode: bool = False,
    pre_action_gate: PreActionGate | None = None,
) -> AgentRuntime:
    """Compose the generic Agent without binding it to Pi0.5 or a camera stack."""
    capabilities = capabilities or CapabilityRegistry()
    selected_controller = controller or DryRunRobotController()
    selected_perception = perception or NullPerceptionProvider()
    if hardware_mode and dry_run:
        raise ValueError(
            "hardware_mode cannot use dry_run policy backends; register real policies."
        )
    if hardware_mode and bool(getattr(planner, "allow_virtual_entities", False)):
        raise ValueError(
            "hardware_mode cannot use simulation virtual entities; build the "
            "scene from physical perception or use a perception-gated planner."
        )
    if hardware_mode and isinstance(selected_controller, DryRunRobotController):
        raise ValueError("hardware_mode requires a real robot controller.")
    if hardware_mode and not bool(
        getattr(selected_controller, "hardware_ready", False)
    ):
        raise ValueError(
            "hardware_mode requires a controller with hardware_ready=True."
        )
    if hardware_mode and isinstance(selected_perception, NullPerceptionProvider):
        raise ValueError("hardware_mode requires a real perception provider.")
    if hardware_mode:
        validate_physical_perception_provider(selected_perception)
    if hardware_mode and (
        verifier is None or isinstance(verifier, PlaceholderActionVerifier)
    ):
        raise ValueError("hardware_mode requires a physical Action Monitor.")
    selected_safety_monitor = safety_monitor or SoftwareSafetyMonitor()
    if hardware_mode and not bool(
        getattr(selected_safety_monitor, "hardware_ready", False)
    ):
        raise ValueError(
            "hardware_mode requires a hardware-approved motion safety profile "
            "with joint limits and live joint telemetry."
        )
    state_provider = AgentStateProvider(
        perception=selected_perception,
        controller=selected_controller,
    )
    selected_policies = policies or PolicyRegistry()
    if hardware_mode:
        dry_run_policies = [
            item["policy_id"]
            for item in selected_policies.describe()
            if isinstance(
                selected_policies.get(str(item["policy_id"])),
                DryRunPolicyBackend,
            )
        ]
        if dry_run_policies:
            raise ValueError(
                "hardware_mode cannot use DryRunPolicyBackend: "
                + ", ".join(dry_run_policies)
            )
    if dry_run:
        for capability in capabilities.planner_skills():
            action_type = capability["action_type"]
            registered = capabilities.get(action_type)
            if registered.executor == "policy" and registered.policy_id:
                if selected_policies.get(registered.policy_id) is None:
                    selected_policies.register(
                        registered.policy_id,
                        DryRunPolicyBackend(),
                        PolicyMetadata(
                            policy_id=registered.policy_id,
                            version="dry-run-1",
                            action_type=action_type,
                            required_inputs=("robot_state",),
                            supports_stop=True,
                            description="Command-only dry-run policy backend.",
                        ),
                    )

    robot_executor = RobotPrimitiveExecutor(selected_controller)
    policy_executor = PolicyExecutor(selected_policies, state_provider, capabilities)
    executors = {"robot": robot_executor, "policy": policy_executor}
    router = ExecutorRouter(
        capabilities,
        executors,
        selected_policies,
        require_policy_registry=True,
    )
    selected_local_recovery = local_recovery or ControllerLocalRecoveryHandler(
        selected_controller,
        state_provider,
    )
    selected_verifier = verifier
    if selected_verifier is None and not isinstance(
        selected_perception, NullPerceptionProvider
    ):
        selected_verifier = StructuredActionMonitor()
    selected_pre_action_gate = pre_action_gate
    if (
        hardware_mode
        and selected_pre_action_gate is not None
        and not isinstance(selected_pre_action_gate, PhysicalTargetGate)
    ):
        raise ValueError(
            "hardware_mode requires PhysicalTargetGate as its pre-action gate."
        )
    if hardware_mode and selected_pre_action_gate is None:
        selected_pre_action_gate = PhysicalTargetGate()
    agent = ClosedLoopAgent(
        planner=planner,
        robot_executor=robot_executor,
        policy_executor=policy_executor,
        state_provider=state_provider,
        verifier=selected_verifier,
        recovery_policy=recovery_policy,
        capabilities=capabilities,
        available_policies=selected_policies.describe(),
        max_replans=max_replans,
        task_store=task_store,
        safety_monitor=selected_safety_monitor,
        task_verifier=task_verifier,
        control=control,
        initial_world_state=initial_world_state,
        router=router,
        local_recovery=selected_local_recovery,
        pre_action_gate=selected_pre_action_gate,
        require_physical_verification=hardware_mode,
    )
    return AgentRuntime(agent, state_provider, capabilities, selected_policies, router)
