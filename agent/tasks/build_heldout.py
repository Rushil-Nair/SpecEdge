"""Write the held-out set of 100 agent tasks (part 6). No model was run on it until the models were final.

    python agent/tasks/build_heldout.py     # writes agent/tasks/heldout_v1.jsonl and heldout_v1.meta.json

Why it exists: the first 100 tasks (tasks_v1.jsonl) were read task by task while studying
model failures, and the third fine-tune is aimed at what was found there. They are now the
development set. This set is what a model is finally judged on.

Same six tools, same checker, same ten families, ten tasks each. Everything a model could
have memorised from the development set is different:
  - people: agent/tools/contacts_heldout.json, twelve new names with ids c_201 to c_212
  - event ids from e_3001; other event titles; dates in February 2027; start times on the
    quarter hour; other lengths, rooms, amounts and expense notes
  - calendars: one to four events on the day in question, one to three on another day
  - wording: in every family, five tasks ask in the development set's words and five in new
    words (field "wording": "as_dev" or "new"), so the effect of wording can be read off

The generator asserts that no question, person, contact id, event id or event title of this
set appears in the development set. Each task has reference_calls and a reference_answer for
verify_tasks.py; they are never shown to a model.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20270211
VERSION = "v1"

CONTACTS = json.loads((HERE.parent / "tools" / "contacts_heldout.json").read_text(encoding="utf-8"))
TEAMS = ["design", "sales", "finance", "engineering"]
TITLES = ["Pipeline review", "Brand workshop", "Supplier call", "Standup", "Tooling demo", "Coffee chat", "Strategy offsite",
          "Interview loop", "Board prep", "Welcome lunch", "Compliance training", "Investor update"]
NEW_TITLES = ["Migration kickoff", "Renewal negotiation", "Wireframe walkthrough", "Invoice audit", "Capacity planning",
              "Incident review", "Partner onboarding", "Launch rehearsal", "Style guide review", "Quota planning",
              "Payroll check", "Usability test debrief"]
DATES = [f"2027-02-{day:02d}" for day in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26)]
STARTS = ["08:15", "08:45", "09:15", "09:45", "10:15", "10:45", "11:15", "11:45", "13:15", "13:45", "14:15", "14:45",
          "15:15", "15:45", "16:15", "16:45"]
DURATIONS = [0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 3.0]
DECIMAL_DURATIONS = [0.25, 0.5, 1.5, 1.75, 2.25]
ROOMS = ["G04", "H118", "K2", "M305", "N41", "P17"]
CATEGORIES = ["travel", "meals", "equipment", "software", "other"]
EXPENSE_NOTES = {"travel": ["airport shuttle", "ferry tickets", "bike rental"],
                 "meals": ["supplier lunch", "conference snacks", "late dinner"],
                 "equipment": ["webcam", "keyboard tray", "whiteboard markers"],
                 "software": ["prototyping seat", "password manager", "stock photos"],
                 "other": ["locker fee", "visa photos", "donation box"]}

# For every family: the development set's wording, then a new one. Same request either way.
WORDING = {
    "schedule_one": ['Schedule a meeting called "{title}" with {person} on {date} at {start} for {hours}.',
                     'I need {hours} with {person} on {date}, starting at {start}. Put it on my calendar as "{title}".'],
    "schedule_team": ['Set up "{title}" on {date} at {start}, {hours} long, and invite everyone on the {team} team.',
                      'Please add "{title}" to my calendar for {date} at {start} ({hours}). The whole {team} team should be invited.'],
    "schedule_two": ['Book "{title}" with {first} and {second} on {date} at {start}. It should last {hours}.',
                     'Can you put "{title}" in the calendar on {date} at {start} for {hours}? Attendees: {first} and {second}.'],
    "cancel_notify": ['Cancel "{title}" on {date} and send {person} a message saying it is cancelled.',
                      '"{title}" on {date} is off. Remove it from my calendar and tell {person} it has been cancelled.'],
    "move": ['Move "{title}" on {date} to {start}. Keep the same date, length and attendees.',
             'On {date}, "{title}" needs to start at {start} instead. Nothing else about it should change.'],
    "notify_attendees": ['Message everyone attending "{title}" on {date} that it will now be held in room {room}.',
                         'The room for "{title}" on {date} has changed to {room}. Let each attendee know by message.'],
    "cancel_fee": ['Cancel "{title}" on {date}, then log the cancellation fee of {amount} as a {category} expense.',
                   'Take "{title}" on {date} off my calendar. It costs a {amount} cancellation fee: record that under {category}.'],
    "conditional_cancel": ["Look at my calendar for {date}. If I have more than two meetings that day, cancel the shortest one; "
                           "otherwise leave it as it is.",
                           "Three or more meetings on {date} is too many for me. If that is the case, drop the shortest of them; "
                           "if not, change nothing."],
    "lookup": ['What time does "{title}" start on {date}?',
               'When is "{title}" on {date}? I need the start time.'],
    "expenses_report": ['Log two expenses: {first} for {cat_1} ("{note_1}") and {second} for {cat_2} ("{note_2}"). '
                        'Then message {person} the combined total.',
                        'I spent {first} on {cat_1} ("{note_1}") and {second} on {cat_2} ("{note_2}"). Record both, then send '
                        '{person} a message with the sum of the two.'],
}


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
        self.next_event = 3001
        self.tasks: list[dict] = []
        self.dates: dict[str, list[str]] = {}

    def date(self, family: str) -> str:
        if family not in self.dates:
            self.dates[family] = self.rng.sample(DATES, len(DATES))
        return self.dates[family].pop()

    def calendar(self, date: str, count: int) -> list[dict]:
        """`count` events on `date`, with different titles, start times and lengths, plus one to
        three events on another date."""
        rng = self.rng
        others = rng.randint(1, 3)
        titles = rng.sample(TITLES, count + others)
        starts = sorted(rng.sample(STARTS, count))
        lengths = rng.sample(DURATIONS, count)
        events = [self.event(title, date, start, hours) for title, start, hours in zip(titles, starts, lengths)]
        other = rng.choice([d for d in DATES if d != date])
        for title, start in zip(titles[count:], rng.sample(STARTS, others)):
            events.append(self.event(title, other, start, rng.choice(DURATIONS)))
        rng.shuffle(events)
        return events

    def event(self, title: str, date: str, start: str, hours: float) -> dict:
        attendees = sorted(c["contact_id"] for c in self.rng.sample(CONTACTS, self.rng.randint(1, 4)))
        event = {"event_id": f"e_{self.next_event}", "title": title, "date": date, "start_time": start,
                 "duration_hours": hours, "attendee_ids": attendees}
        self.next_event += 1
        return event

    @staticmethod
    def kept(events: list[dict]) -> list[dict]:
        return [{k: v for k, v in event.items() if k != "event_id"} for event in events]

    def free_start(self, events: list[dict]) -> str:
        return self.rng.choice([s for s in STARTS if s not in {e["start_time"] for e in events}])

    def add(self, family: str, fields: dict, initial: list[dict], expect: dict, reference: list[dict],
            answer: str = "Done.") -> None:
        number = sum(task["family"] == family for task in self.tasks) + 1
        wording = (number - 1) % 2                      # odd-numbered tasks: the development wording
        expect.setdefault("expenses", [])
        expect.setdefault("messages", [])
        self.tasks.append({"id": f"h_{family}_{number:02d}", "family": family, "wording": ["as_dev", "new"][wording],
                           "user": WORDING[family][wording].format(**fields), "initial": {"events": initial},
                           "expect": expect, "reference_calls": reference, "reference_answer": answer})

    # --- the ten families ---------------------------------------------------------------

    def schedule_one(self) -> None:
        rng = self.rng
        person, date = rng.choice(CONTACTS), self.date("schedule_one")
        initial = self.calendar(date, rng.randint(1, 3))
        new = {"title": rng.choice(NEW_TITLES), "date": date, "start_time": self.free_start(initial),
               "duration_hours": rng.choice(DECIMAL_DURATIONS), "attendee_ids": [person["contact_id"]]}
        self.add("schedule_one", {"title": new["title"], "person": person["name"], "date": date, "start": new["start_time"],
                                  "hours": hours_text(new["duration_hours"])},
                 initial, {"events": self.kept(initial) + [new], "required_tools": ["search_contacts", "create_event"]},
                 [{"name": "search_contacts", "arguments": {"query": person["name"]}},
                  {"name": "create_event", "arguments": new}])

    def schedule_team(self) -> None:
        rng = self.rng
        team, date = rng.choice(TEAMS), self.date("schedule_team")
        initial = self.calendar(date, rng.randint(1, 3))
        new = {"title": rng.choice(NEW_TITLES), "date": date, "start_time": self.free_start(initial),
               "duration_hours": rng.choice(DECIMAL_DURATIONS),
               "attendee_ids": sorted(c["contact_id"] for c in CONTACTS if c["team"] == team)}
        self.add("schedule_team", {"title": new["title"], "team": team, "date": date, "start": new["start_time"],
                                   "hours": hours_text(new["duration_hours"])},
                 initial, {"events": self.kept(initial) + [new], "required_tools": ["search_contacts", "create_event"]},
                 [{"name": "search_contacts", "arguments": {"query": team}},
                  {"name": "create_event", "arguments": new}])

    def schedule_two(self) -> None:
        rng = self.rng
        first, second = rng.sample(CONTACTS, 2)
        date = self.date("schedule_two")
        initial = self.calendar(date, rng.randint(1, 3))
        new = {"title": rng.choice(NEW_TITLES), "date": date, "start_time": self.free_start(initial),
               "duration_hours": rng.choice(DECIMAL_DURATIONS),
               "attendee_ids": sorted([first["contact_id"], second["contact_id"]])}
        self.add("schedule_two", {"title": new["title"], "first": first["name"], "second": second["name"], "date": date,
                                  "start": new["start_time"], "hours": hours_text(new["duration_hours"])},
                 initial, {"events": self.kept(initial) + [new], "required_tools": ["search_contacts", "create_event"]},
                 [{"name": "search_contacts", "arguments": {"query": first["name"]}},
                  {"name": "search_contacts", "arguments": {"query": second["name"]}},
                  {"name": "create_event", "arguments": new}])

    def cancel_notify(self) -> None:
        rng = self.rng
        date = self.date("cancel_notify")
        initial = self.calendar(date, rng.randint(2, 4))
        target = rng.choice([e for e in initial if e["date"] == date])
        person = rng.choice(CONTACTS)
        self.add("cancel_notify", {"title": target["title"], "date": date, "person": person["name"]}, initial,
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
        initial = self.calendar(date, rng.randint(2, 4))
        target = rng.choice([e for e in initial if e["date"] == date])
        moved = dict(target, start_time=self.free_start(initial))
        self.add("move", {"title": target["title"], "date": date, "start": moved["start_time"]}, initial,
                 {"events": self.kept([moved if e is target else e for e in initial]), "required_tools": ["list_events"]},
                 [{"name": "list_events", "arguments": {"date": date}},
                  {"name": "cancel_event", "arguments": {"event_id": target["event_id"]}},
                  {"name": "create_event", "arguments": {k: v for k, v in moved.items() if k != "event_id"}}])

    def notify_attendees(self) -> None:
        rng = self.rng
        date = self.date("notify_attendees")
        initial = self.calendar(date, rng.randint(2, 4))
        target = rng.choice([e for e in initial if e["date"] == date])
        room = rng.choice(ROOMS)
        self.add("notify_attendees", {"title": target["title"], "date": date, "room": room}, initial,
                 {"events": self.kept(initial),
                  "messages": [{"contact_id": c, "contains": [room]} for c in target["attendee_ids"]],
                  "required_tools": ["list_events", "send_message"]},
                 [{"name": "list_events", "arguments": {"date": date}}] +
                 [{"name": "send_message", "arguments": {"contact_id": c, "text": f'{target["title"]} moves to room {room}.'}}
                  for c in target["attendee_ids"]])

    def cancel_fee(self) -> None:
        rng = self.rng
        date = self.date("cancel_fee")
        initial = self.calendar(date, rng.randint(2, 4))
        target = rng.choice([e for e in initial if e["date"] == date])
        amount = round(rng.randint(6, 140) + rng.choice([0.05, 0.3, 0.45, 0.65, 0.7, 0.8, 0.9, 0.35]), 2)
        category = rng.choice(CATEGORIES)
        self.add("cancel_fee", {"title": target["title"], "date": date, "amount": money(amount), "category": category},
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
        self.add("conditional_cancel", {"date": date}, initial,
                 {"events": self.kept(final), "required_tools": ["list_events"]}, reference)

    def lookup(self) -> None:
        rng = self.rng
        date = self.date("lookup")
        while True:
            initial = self.calendar(date, rng.randint(2, 4))
            today = [e for e in initial if e["date"] == date]
            target = rng.choice(today)
            # The checker looks for "13:15" or "1:15" in the answer. No other event that day may
            # start at a time containing either ("11:15"), or a wrong answer could pass.
            forms = (target["start_time"], twelve_hour(target["start_time"]))
            if not any(form in e["start_time"] for e in today if e is not target for form in forms):
                break
        self.add("lookup", {"title": target["title"], "date": date}, initial,
                 {"events": self.kept(initial),
                  "answer_contains": [[target["start_time"], twelve_hour(target["start_time"])]],
                  "required_tools": ["list_events"]},
                 [{"name": "list_events", "arguments": {"date": date}}],
                 answer=f'{target["title"]} starts at {target["start_time"]}.')

    def expenses_report(self) -> None:
        rng = self.rng
        person = rng.choice(CONTACTS)
        initial = self.calendar(rng.choice(DATES), rng.randint(1, 3))
        while True:
            first = round(rng.randint(4, 90) + rng.choice([0.15, 0.3, 0.55, 0.65, 0.85, 0.95]), 2)
            second = round(rng.randint(4, 90) + rng.choice([0.1, 0.2, 0.45, 0.5, 0.7, 0.99]), 2)
            total = round(first + second, 2)
            if round(total * 100) % 10:            # the total keeps two real decimals
                break
        cat_1, cat_2 = rng.sample(CATEGORIES, 2)
        note_1, note_2 = rng.choice(EXPENSE_NOTES[cat_1]), rng.choice(EXPENSE_NOTES[cat_2])
        self.add("expenses_report", {"first": money(first), "cat_1": cat_1, "note_1": note_1, "second": money(second),
                                     "cat_2": cat_2, "note_2": note_2, "person": person["name"]},
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


def check_apart_from_dev(tasks: list[dict]) -> dict:
    """Nothing a model could have memorised from the development set may appear here."""
    dev = [json.loads(line) for line in (HERE / "tasks_v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_contacts = json.loads((HERE.parent / "tools" / "contacts.json").read_text(encoding="utf-8"))

    def facts(rows: list[dict], contacts: list[dict]) -> dict[str, set]:
        events = [e for row in rows for e in row["initial"]["events"]] + \
                 [c["arguments"] for row in rows for c in row["reference_calls"] if c["name"] == "create_event"]
        return {"questions": {row["user"] for row in rows},
                "people": {c["name"] for c in contacts} | {part for c in contacts for part in c["name"].split()},
                "contact ids": {c["contact_id"] for c in contacts},
                "event ids": {e["event_id"] for row in rows for e in row["initial"]["events"]},
                "event titles": {e["title"] for e in events},
                "dates": {e["date"] for e in events}}

    mine, theirs = facts(tasks, CONTACTS), facts(dev, dev_contacts)
    shared = {name: sorted(mine[name] & theirs[name]) for name in mine}
    assert not any(shared.values()), f"shared with the development set: {shared}"
    return {name: 0 for name in mine}


def main() -> None:
    names = [c["name"].lower() for c in CONTACTS]
    assert all(a == b or a not in b for a in names for b in names), "one name contains another"
    assert len({part for name in names for part in name.split()}) == sum(len(name.split()) for name in names), "a name part is used twice"

    maker = Maker()
    for index in range(10):
        maker.schedule_one()
        maker.schedule_team()
        maker.schedule_two()
        maker.cancel_notify()
        maker.move()
        maker.notify_attendees()
        maker.cancel_fee()
        maker.conditional_cancel(crowded=index % 4 < 2)     # 6 crowded, 4 not; both wordings get both kinds
        maker.lookup()
        maker.expenses_report()
    tasks = sorted(maker.tasks, key=lambda task: task["id"])
    assert len(tasks) == 100 and len({task["id"] for task in tasks}) == 100
    assert len({task["user"] for task in tasks}) == 100, "two tasks ask the same thing"
    shared = check_apart_from_dev(tasks)

    out = HERE / f"heldout_{VERSION}.jsonl"
    text = "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks)
    out.write_bytes(text.encode("utf-8"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    calls = [len(task["reference_calls"]) for task in tasks]
    contacts_digest = hashlib.sha256((HERE.parent / "tools" / "contacts_heldout.json").read_bytes()).hexdigest()
    (HERE / f"heldout_{VERSION}.meta.json").write_text(json.dumps({
        "set": "held-out", "version": VERSION, "seed": SEED, "tasks": len(tasks), "sha256": digest,
        "contacts_file": "agent/tools/contacts_heldout.json", "contacts_sha256": contacts_digest,
        "families": sorted({task["family"] for task in tasks}), "tasks_per_family": 10,
        "wording": {"as_dev": sum(t["wording"] == "as_dev" for t in tasks), "new": sum(t["wording"] == "new" for t in tasks)},
        "reference_calls_per_task": {"min": min(calls), "max": max(calls), "mean": sum(calls) / len(calls)},
        "shared_with_development_set": shared,
        "rule": "No model is run on this set until the models are final. Results on it are reported apart from the development set.",
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(tasks)} tasks -> {out.name}  sha256 {digest}")
    print(f"reference calls per task: min {min(calls)}, max {max(calls)}, mean {sum(calls) / len(calls):.2f}")
    print("shared with the development set:", shared)


if __name__ == "__main__":
    main()
