"""Write the 100 scripted multi-step tasks for the agent runtime (part 6).

    python agent/tasks/build_tasks.py            # writes agent/tasks/tasks_v1.jsonl and its checksum

Ten families, ten tasks each, generated from a fixed seed. Every task takes at least two
model turns: something the model must do or say depends on what a tool returned (an id, a
start time, a list of attendees). The reference solutions use 1 to 4 calls. Each task carries:

  user             what the user asks
  initial          the calendar the task starts from
  expect           the state the task must end in, for agent/tools' checker:
                     events, expenses, messages (compared as whole sets: nothing missing,
                     nothing extra), answer_contains (words the final reply must contain),
                     required_tools (tools that must have been called)
  reference_calls  one correct sequence of calls. Used only by verify_tasks.py to prove the
                   task can be solved and that the checker accepts the solution; never shown
                   to a model.

The contacts are the fixed directory in agent/tools/contacts.json.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261008
VERSION = "v1"

CONTACTS = json.loads((HERE.parent / "tools" / "contacts.json").read_text(encoding="utf-8"))
TEAMS = ["design", "sales", "finance", "engineering"]
TITLES = ["Budget review", "Design sync", "Client call", "Sprint planning", "Vendor demo", "Team lunch",
          "Roadmap workshop", "Hiring panel", "Quarterly review", "Onboarding session", "Security briefing",
          "Product demo"]
NEW_TITLES = ["Launch prep", "Pricing discussion", "Design critique", "Contract review", "Data migration check",
              "Offsite planning", "Customer feedback session", "Release retro", "Brand refresh kickoff",
              "Support handover", "Forecast walkthrough", "Accessibility audit"]
DATES = [f"2026-11-{day:02d}" for day in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27)]
STARTS = ["08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "13:00", "13:30", "14:00", "14:30",
          "15:00", "15:30", "16:00", "16:30"]
DURATIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
DECIMAL_DURATIONS = [0.5, 0.75, 1.25, 1.5, 2.5]
ROOMS = ["B12", "C204", "A7", "D310", "E15", "F221"]
CATEGORIES = ["travel", "meals", "equipment", "software", "other"]
EXPENSE_NOTES = {"travel": ["taxi to the airport", "train tickets", "parking"],
                 "meals": ["client dinner", "team breakfast", "working lunch"],
                 "equipment": ["monitor stand", "USB hub", "desk lamp"],
                 "software": ["diagram plugin", "font licence", "survey tool"],
                 "other": ["courier", "printing", "gift card"]}


def hours_text(hours: float) -> str:
    return f"{hours:g} hour" + ("" if hours == 1 else "s")


def money(amount: float) -> str:
    return f"{amount:.2f}"


def twelve_hour(start: str) -> str:
    hour, minute = int(start[:2]), start[3:]
    return f"{(hour - 1) % 12 + 1}:{minute}"


class Maker:
    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.next_event = 101
        self.tasks: list[dict] = []
        self.dates: dict[str, list[str]] = {}

    def date(self, family: str) -> str:
        """A date no other task of this family uses, so no two tasks ask the same thing."""
        if family not in self.dates:
            self.dates[family] = self.rng.sample(DATES, len(DATES))
        return self.dates[family].pop()

    def calendar(self, date: str, count: int, other_date: bool = True) -> list[dict]:
        """`count` events on `date` with different titles, start times and lengths, plus two
        events on another date as distractors."""
        rng = self.rng
        events = []
        titles = rng.sample(TITLES, count + 2)
        starts = sorted(rng.sample(STARTS, count))
        lengths = rng.sample(DURATIONS, count)
        for title, start, hours in zip(titles, starts, lengths):
            events.append(self.event(title, date, start, hours))
        if other_date:
            other = rng.choice([d for d in DATES if d != date])
            for title in titles[count:]:
                events.append(self.event(title, other, rng.choice(STARTS), rng.choice(DURATIONS)))
        rng.shuffle(events)
        return events

    def event(self, title: str, date: str, start: str, hours: float) -> dict:
        attendees = sorted(c["contact_id"] for c in self.rng.sample(CONTACTS, self.rng.randint(1, 3)))
        event = {"event_id": f"e_{self.next_event}", "title": title, "date": date, "start_time": start,
                 "duration_hours": hours, "attendee_ids": attendees}
        self.next_event += 1
        return event

    @staticmethod
    def kept(events: list[dict]) -> list[dict]:
        """Events as the checker compares them: without ids."""
        return [{k: v for k, v in event.items() if k != "event_id"} for event in events]

    def add(self, family: str, user: str, initial: list[dict], expect: dict, reference: list[dict]) -> None:
        number = sum(task["family"] == family for task in self.tasks) + 1
        expect.setdefault("expenses", [])
        expect.setdefault("messages", [])
        self.tasks.append({"id": f"{family}_{number:02d}", "family": family, "user": user,
                           "initial": {"events": initial}, "expect": expect, "reference_calls": reference})

    # --- the ten families ---------------------------------------------------------------

    def schedule_one(self) -> None:
        rng = self.rng
        person, date = rng.choice(CONTACTS), self.date("schedule_one")
        initial = self.calendar(date, 2)
        title, hours = rng.choice(NEW_TITLES), rng.choice(DECIMAL_DURATIONS)
        start = rng.choice([s for s in STARTS if s not in {e["start_time"] for e in initial}])
        new = {"title": title, "date": date, "start_time": start, "duration_hours": hours,
               "attendee_ids": [person["contact_id"]]}
        self.add("schedule_one",
                 f'Schedule a meeting called "{title}" with {person["name"]} on {date} at {start} for {hours_text(hours)}.',
                 initial, {"events": self.kept(initial) + [new], "required_tools": ["search_contacts", "create_event"]},
                 [{"name": "search_contacts", "arguments": {"query": person["name"]}},
                  {"name": "create_event", "arguments": new}])

    def schedule_team(self) -> None:
        rng = self.rng
        team, date = rng.choice(TEAMS), self.date("schedule_team")
        members = sorted(c["contact_id"] for c in CONTACTS if c["team"] == team)
        initial = self.calendar(date, 2)
        title, hours = rng.choice(NEW_TITLES), rng.choice(DECIMAL_DURATIONS)
        start = rng.choice([s for s in STARTS if s not in {e["start_time"] for e in initial}])
        new = {"title": title, "date": date, "start_time": start, "duration_hours": hours, "attendee_ids": members}
        self.add("schedule_team",
                 f'Set up "{title}" on {date} at {start}, {hours_text(hours)} long, and invite everyone on the {team} team.',
                 initial, {"events": self.kept(initial) + [new], "required_tools": ["search_contacts", "create_event"]},
                 [{"name": "search_contacts", "arguments": {"query": team}},
                  {"name": "create_event", "arguments": new}])

    def schedule_two(self) -> None:
        rng = self.rng
        first, second = rng.sample(CONTACTS, 2)
        date = self.date("schedule_two")
        initial = self.calendar(date, 2)
        title, hours = rng.choice(NEW_TITLES), rng.choice(DECIMAL_DURATIONS)
        start = rng.choice([s for s in STARTS if s not in {e["start_time"] for e in initial}])
        new = {"title": title, "date": date, "start_time": start, "duration_hours": hours,
               "attendee_ids": sorted([first["contact_id"], second["contact_id"]])}
        self.add("schedule_two",
                 f'Book "{title}" with {first["name"]} and {second["name"]} on {date} at {start}. It should last {hours_text(hours)}.',
                 initial, {"events": self.kept(initial) + [new], "required_tools": ["search_contacts", "create_event"]},
                 [{"name": "search_contacts", "arguments": {"query": first["name"]}},
                  {"name": "search_contacts", "arguments": {"query": second["name"]}},
                  {"name": "create_event", "arguments": new}])

    def cancel_notify(self) -> None:
        rng = self.rng
        date = self.date("cancel_notify")
        initial = self.calendar(date, 3)
        target = rng.choice([e for e in initial if e["date"] == date])
        person = rng.choice(CONTACTS)
        self.add("cancel_notify",
                 f'Cancel "{target["title"]}" on {date} and send {person["name"]} a message saying it is cancelled.',
                 initial,
                 {"events": self.kept([e for e in initial if e is not target]),
                  "messages": [{"contact_id": person["contact_id"], "contains": ["cancel"]}],
                  "required_tools": ["list_events", "cancel_event", "search_contacts", "send_message"]},
                 [{"name": "list_events", "arguments": {"date": date}},
                  {"name": "cancel_event", "arguments": {"event_id": target["event_id"]}},
                  {"name": "search_contacts", "arguments": {"query": person["name"]}},
                  {"name": "send_message", "arguments": {"contact_id": person["contact_id"],
                                                         "text": f'{target["title"]} on {date} is cancelled.'}}])

    def move(self) -> None:
        rng = self.rng
        date = self.date("move")
        initial = self.calendar(date, 3)
        target = rng.choice([e for e in initial if e["date"] == date])
        start = rng.choice([s for s in STARTS if s not in {e["start_time"] for e in initial}])
        moved = dict(target, start_time=start)
        self.add("move",
                 f'Move "{target["title"]}" on {date} to {start}. Keep the same date, length and attendees.',
                 initial,
                 {"events": self.kept([moved if e is target else e for e in initial]),
                  "required_tools": ["list_events"]},
                 [{"name": "list_events", "arguments": {"date": date}},
                  {"name": "cancel_event", "arguments": {"event_id": target["event_id"]}},
                  {"name": "create_event", "arguments": {k: v for k, v in moved.items() if k != "event_id"}}])

    def notify_attendees(self) -> None:
        rng = self.rng
        date = self.date("notify_attendees")
        initial = self.calendar(date, 3)
        target = rng.choice([e for e in initial if e["date"] == date])
        room = rng.choice(ROOMS)
        self.add("notify_attendees",
                 f'Message everyone attending "{target["title"]}" on {date} that it will now be held in room {room}.',
                 initial,
                 {"events": self.kept(initial),
                  "messages": [{"contact_id": c, "contains": [room]} for c in target["attendee_ids"]],
                  "required_tools": ["list_events", "send_message"]},
                 [{"name": "list_events", "arguments": {"date": date}}] +
                 [{"name": "send_message", "arguments": {"contact_id": c, "text": f'{target["title"]} moves to room {room}.'}}
                  for c in target["attendee_ids"]])

    def cancel_fee(self) -> None:
        rng = self.rng
        date = self.date("cancel_fee")
        initial = self.calendar(date, 3)
        target = rng.choice([e for e in initial if e["date"] == date])
        amount = round(rng.randint(8, 95) + rng.choice([0.25, 0.4, 0.5, 0.6, 0.75, 0.95, 0.15, 0.85]), 2)
        category = rng.choice(CATEGORIES)
        self.add("cancel_fee",
                 f'Cancel "{target["title"]}" on {date}, then log the cancellation fee of {money(amount)} as a {category} expense.',
                 initial,
                 {"events": self.kept([e for e in initial if e is not target]),
                  "expenses": [{"amount": amount, "category": category}],
                  "required_tools": ["list_events", "cancel_event", "add_expense"]},
                 [{"name": "list_events", "arguments": {"date": date}},
                  {"name": "cancel_event", "arguments": {"event_id": target["event_id"]}},
                  {"name": "add_expense", "arguments": {"amount": amount, "category": category, "note": "Cancellation fee"}}])

    def conditional_cancel(self, crowded: bool) -> None:
        rng = self.rng
        date = self.date("conditional_cancel")
        initial = self.calendar(date, rng.choice([3, 4]) if crowded else rng.choice([1, 2]))
        today = [e for e in initial if e["date"] == date]
        reference = [{"name": "list_events", "arguments": {"date": date}}]
        final = initial
        if crowded:
            shortest = min(today, key=lambda e: e["duration_hours"])      # lengths on one date are all different
            final = [e for e in initial if e is not shortest]
            reference.append({"name": "cancel_event", "arguments": {"event_id": shortest["event_id"]}})
        self.add("conditional_cancel",
                 f"Look at my calendar for {date}. If I have more than two meetings that day, cancel the shortest one; "
                 "otherwise leave it as it is.",
                 initial, {"events": self.kept(final), "required_tools": ["list_events"]}, reference)

    def lookup(self) -> None:
        rng = self.rng
        date = self.date("lookup")
        initial = self.calendar(date, 3)
        target = rng.choice([e for e in initial if e["date"] == date])
        self.add("lookup",
                 f'What time does "{target["title"]}" start on {date}?',
                 initial,
                 {"events": self.kept(initial),
                  "answer_contains": [[target["start_time"], twelve_hour(target["start_time"])]],
                  "required_tools": ["list_events"]},
                 [{"name": "list_events", "arguments": {"date": date}}])
        self.tasks[-1]["reference_answer"] = f'{target["title"]} starts at {target["start_time"]}.'

    def expenses_report(self) -> None:
        rng = self.rng
        person = rng.choice(CONTACTS)
        initial = self.calendar(rng.choice(DATES), 2)
        while True:
            first = round(rng.randint(5, 60) + rng.choice([0.2, 0.35, 0.45, 0.6, 0.75, 0.99]), 2)
            second = round(rng.randint(5, 60) + rng.choice([0.15, 0.25, 0.4, 0.55, 0.8, 0.9]), 2)
            total = round(first + second, 2)
            if round(total * 100) % 10:            # the total has two real decimals, so "33.39" must appear as such
                break
        cat_1, cat_2 = rng.sample(CATEGORIES, 2)
        note_1, note_2 = rng.choice(EXPENSE_NOTES[cat_1]), rng.choice(EXPENSE_NOTES[cat_2])
        self.add("expenses_report",
                 f'Log two expenses: {money(first)} for {cat_1} ("{note_1}") and {money(second)} for {cat_2} ("{note_2}"). '
                 f'Then message {person["name"]} the combined total.',
                 initial,
                 {"events": self.kept(initial),
                  "expenses": [{"amount": first, "category": cat_1}, {"amount": second, "category": cat_2}],
                  "messages": [{"contact_id": person["contact_id"], "contains": [money(total)]}],
                  "required_tools": ["add_expense", "search_contacts", "send_message"]},
                 [{"name": "add_expense", "arguments": {"amount": first, "category": cat_1, "note": note_1}},
                  {"name": "add_expense", "arguments": {"amount": second, "category": cat_2, "note": note_2}},
                  {"name": "search_contacts", "arguments": {"query": person["name"]}},
                  {"name": "send_message", "arguments": {"contact_id": person["contact_id"],
                                                         "text": f"The combined total is {money(total)}."}}])


def main() -> None:
    maker = Maker()
    for index in range(10):
        maker.schedule_one()
        maker.schedule_team()
        maker.schedule_two()
        maker.cancel_notify()
        maker.move()
        maker.notify_attendees()
        maker.cancel_fee()
        maker.conditional_cancel(crowded=index % 2 == 0)
        maker.lookup()
        maker.expenses_report()
    tasks = sorted(maker.tasks, key=lambda task: task["id"])
    for task in tasks:
        task.setdefault("reference_answer", "Done.")
    assert len(tasks) == 100 and len({task["id"] for task in tasks}) == 100
    assert len({task["user"] for task in tasks}) == 100, "two tasks ask the same thing"

    out = HERE / f"tasks_{VERSION}.jsonl"
    text = "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks)
    out.write_bytes(text.encode("utf-8"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    calls = [len(task["reference_calls"]) for task in tasks]
    (HERE / f"tasks_{VERSION}.meta.json").write_text(json.dumps({
        "version": VERSION, "seed": SEED, "tasks": len(tasks), "sha256": digest,
        "families": sorted({task["family"] for task in tasks}), "tasks_per_family": 10,
        "reference_calls_per_task": {"min": min(calls), "max": max(calls), "mean": sum(calls) / len(calls)},
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(tasks)} tasks -> {out.name}  sha256 {digest}")
    print(f"reference calls per task: min {min(calls)}, max {max(calls)}, mean {sum(calls) / len(calls):.2f}")


if __name__ == "__main__":
    main()
