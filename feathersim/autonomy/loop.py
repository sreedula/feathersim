"""The unattended machine-tending autonomy loop — the headline deliverable. [Phase 5]

`run_autonomy` composes everything below it: it reads machine state through **perception** (the model's
prediction from rendered pixels, never the sim's ground truth), picks whichever machine is *perceived*
``done``, and tends it through the skill SDK (navigate → unload → carry → place). Repeat, unattended,
for a target number of parts. It deliberately trusts perception: if a reading was a false positive the
downstream skill's precondition raises :class:`~feathersim.sdk.robot.PreconditionError`, which the loop
catches and shrugs off — while a genuine navigation failure (plain ``SkillError``) is left to propagate.

The split that makes this honest: selection consumes :meth:`Perception.perceive` only; the SDK's
ground-truth accessors are never used to decide *what* to tend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from feathersim.sdk.robot import PreconditionError, Robot
from feathersim.sim.machine import MachineState
from feathersim.sim.world import TIMESTEP, World

if TYPE_CHECKING:
    import mujoco

    from feathersim.perception.infer import PerceivedState


class Perceiver(Protocol):
    """The perception seam the loop selects on: predictions from pixels, keyed by machine name.

    Typing the collaborator (rather than importing the concrete :class:`Perception`) documents the
    perceived-vs-ground-truth invariant — the loop consumes *only* this — and lets a test double
    stand in. The real :class:`feathersim.perception.infer.Perception` satisfies it structurally.
    """

    def perceive(self, world: World, renderer: mujoco.Renderer | None) -> dict[str, PerceivedState]:
        ...


@dataclass(frozen=True)
class TendEvent:
    """One completed tend: which machine, the part delivered, sim time, and the perception confidence
    that triggered it."""

    machine: str
    part: str
    sim_time: float
    confidence: float


@dataclass
class AutonomyReport:
    """Outcome of an autonomy run — the throughput/uptime log the acceptance asks for."""

    parts_delivered: int
    sim_seconds: float
    per_machine: dict[str, int]
    events: list[TendEvent] = field(default_factory=list)

    @property
    def throughput_per_min(self) -> float:
        """Parts delivered per sim minute of uptime (0 if no time has elapsed)."""
        return 0.0 if self.sim_seconds <= 0 else self.parts_delivered * 60.0 / self.sim_seconds


def _advance(world, seconds: float) -> None:
    """Step the sim ~``seconds`` forward so machines keep cycling while the robot waits/idles."""
    for _ in range(max(1, round(seconds / TIMESTEP))):
        world.step()


def run_autonomy(
    world: World,
    perception: Perceiver,
    renderer: mujoco.Renderer | None,
    *,
    target_parts: int = 6,
    max_sim_seconds: float = 600.0,
    wait_seconds: float = 0.5,
    min_confidence: float = 0.0,
    robot: Robot | None = None,
    on_event: Callable[[TendEvent], None] | None = None,
) -> AutonomyReport:
    """Run the unattended tend loop until ``target_parts`` are delivered (or the sim-time budget runs out).

    Selection is driven purely by ``perception.perceive(world, renderer)``: among the machines
    *perceived* ``done`` (with confidence ≥ ``min_confidence``), the one that has been waiting longest is
    tended next (oldest-first — fair and throughput-maximizing, since a done machine has stopped cycling).
    With nothing perceived done, the sim advances ``wait_seconds`` and re-perceives.

    Robustness: a perception **false positive** surfaces as a :class:`PreconditionError` from the SDK
    (the machine wasn't actually ``done``); the loop drops that candidate and falls through, so a bad
    reading can't crash or starve a genuinely-done machine. A genuine **navigation** failure raises a
    plain :class:`~feathersim.sdk.robot.SkillError` (not a precondition error) and is deliberately *not*
    swallowed — an unattended run should fail loudly on a real nav regression, not silently lose throughput.

    Every iteration makes progress (a tend advances sim time; otherwise ``_advance`` does), so the run
    is bounded by ``target_parts`` and the ``max_sim_seconds`` budget. The budget is a *soft* cap —
    checked between iterations, so an in-flight ``tend`` may overshoot it by one trip.
    """
    robot = robot if robot is not None else Robot(world)
    per_machine: dict[str, int] = {m.name: 0 for m in world.machines}
    events: list[TendEvent] = []
    # Sim time each machine was *first* perceived done in its current done episode — the basis for
    # oldest-waiting-first scheduling. A machine that's been done longest is the most blocked (it
    # stopped cycling when it finished), so servicing it first is both starvation-free and the
    # throughput-maximizing order. Cleared when a machine stops reading done (serviced or false read).
    done_since: dict[str, float] = {}

    while len(events) < target_parts and world.time < max_sim_seconds:
        readings = perception.perceive(world, renderer)
        now = world.time
        perceived_done = {
            n for n, r in readings.items()
            if r.machine_state is MachineState.DONE and r.confidence >= min_confidence
        }
        done_since = {n: t for n, t in done_since.items() if n in perceived_done}  # drop stale
        for n in perceived_done:
            done_since.setdefault(n, now)  # stamp newly-observed done machines
        # Longest-waiting first; deterministic tie-break by name for machines stamped the same tick.
        candidates = sorted(perceived_done, key=lambda n: (done_since[n], n))

        tended = False
        for name in candidates:
            try:
                part = robot.tend(name)
            except PreconditionError:
                # False positive: perception said done but the precondition disagreed. Drop its timer
                # and fall through, so one bad reading can't hold priority or starve a real one. A real
                # navigation failure raises plain SkillError instead and is intentionally not caught.
                done_since.pop(name, None)
                continue
            per_machine[name] += 1
            event = TendEvent(name, part, world.time, readings[name].confidence)
            events.append(event)
            if on_event is not None:
                on_event(event)
            done_since.pop(name, None)  # serviced — its waiting clock resets for the next episode
            tended = True
            break  # state changed (machine reset, others may have finished) — re-perceive

        if not tended:
            _advance(world, wait_seconds)  # nobody actually ready — let the machines run

    return AutonomyReport(
        parts_delivered=len(events),
        sim_seconds=world.time,
        per_machine=per_machine,
        events=events,
    )
