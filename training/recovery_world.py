"""Simulated tool worlds and scripted conversations for the third fine-tune (error recovery).

Used by training/prepare_recovery.py (training examples) and bench/recovery_eval.py (the two
held-out topics, run live against a model). Nothing here needs a model or a tokenizer.

Why. The failure breakdown for `mine_multistep` (runtime/results/part6/
FAILURES_mine_multistep.md): in 54 failed tasks per layout the model put
a name where an id belongs, got a tool error, and in 40 of them stopped on the next turn
without looking the id up. This file writes conversations that show the other behaviour, on
topics and tools that are not the agent tasks' (no calendar, contacts, expenses or messaging):

  lookup_create, lookup_cancel   a name is given: look the id up first, then act with the id
  recover_create, recover_cancel a call failed on a made-up id: look it up, then try again
                                 (the failed call is context only, never a training target)
  move                           "move" or "change": look up, create the new one, cancel the old one
  partial                        two-part request where one part cannot be done: do the other,
                                 then say plainly what was done and what was not
  ask                            the lookup finds nobody or several: only then ask the user
  direct                         the id is given: act without a lookup

Every topic has the same skeleton with its own words: things with names (ITEM) and people or
places with names (PARTY), both of which have ids; records that join one of each and carry
one attribute; a lookup tool for each; create, cancel and list tools for the records; one
side tool that takes free text. Tool names, parameter names, id formats, result layouts and
error wording differ from topic to topic. Tool results come from a small in-memory world, so
every result and every error is the consequence of the calls before it.
"""

from __future__ import annotations

import copy
import json
import random

FIRST = ["Dana", "Jonas", "Imani", "Rafael", "Sofia", "Kenji", "Amara", "Viktor", "Leila", "Mateo", "Ingrid", "Tariq",
         "Noor", "Callum", "Yuki", "Beatriz", "Emeka", "Freya", "Anton", "Carmen", "Idris", "Maeve", "Pavel", "Sana",
         "Thabo", "Elif", "Hugo", "Mirela", "Joon", "Astrid", "Rohan", "Celine", "Kofi", "Nadia", "Stefan", "Wanjiru"]
LAST = ["Okafor", "Lindqvist", "Marchetti", "Haddad", "Petrov", "Yamada", "Abara", "Castillo", "Brandt", "Mwangi", "Duarte",
        "Halloran", "Sato", "Eriksen", "Baptiste", "Kowalski", "Farouk", "Nakamura", "Olsen", "Varga", "Achebe", "Moreau",
        "Ibarra", "Stojanov", "Quint", "Fontaine", "Dlamini", "Hartmann", "Reyes", "Sorensen", "Adeyemi", "Popescu"]
PEOPLE = "people"          # marker: party names are drawn from FIRST x LAST

# attr kinds: date, time, int:<lo>:<hi>, or a list of choices
TOPICS: dict[str, dict] = {
    "library": dict(
        item=("book", "book_id", "b_{n}", ["The Salt Road", "Glass Harbour", "A Field Guide to Moss", "Night Ferry", "The Cartographer's Niece",
                                           "Small Engines", "Winter Orchard", "The Ninth Bridge", "Paper Lanterns", "Slow Rivers"]),
        party=("member", "member_id", "m_{n}", PEOPLE),
        record=("hold", "hold_id", "h_{n}"), attr=("pickup_date", "date", "The date the book will be picked up, as YYYY-MM-DD."),
        tools=dict(find_item=("find_books", "title", "Search the catalogue by title or part of a title. Returns each matching book with its book id."),
                   find_party=("find_members", "name", "Search library members by name. Returns each match with its member id."),
                   list=("list_holds", "List the holds one member has, with hold ids, book ids, titles and pickup dates."),
                   create=("place_hold", "Place a hold on a book for a member."),
                   cancel=("release_hold", "Release one hold."),
                   side=("add_desk_note", "text", "Leave a note for the front desk.")),
        ask=dict(create=['place a hold on "{item}" for {party}, to be picked up on {attr}', 'hold "{item}" for {party} with pickup on {attr}'],
                 cancel=['release {party}\'s hold on "{item}"', 'drop the hold that {party} has on "{item}"'],
                 move=['move {party}\'s hold on "{item}" to a pickup on {new}', 'change the pickup date of {party}\'s hold on "{item}" to {new}'],
                 side=['add a desk note saying "{text}"', 'leave the front desk a note: "{text}"']),
        notes=["member will collect after 5 pm", "call before shelving", "damaged cover reported"], style=0),
    "warehouse": dict(
        item=("product", "product_id", "SKU-{n}", ["steel shelf bracket", "nitrile gloves size M", "pallet wrap 500 mm", "LED batten 1.2 m", "cable tie 300 mm",
                                                   "hex bolt M8", "thermal label roll", "safety goggles", "floor marking tape", "corner protector"]),
        party=("site", "site_id", "WH{n}", ["Northgate depot", "Riverside hub", "Eastfield store", "Harbour annex", "Millbrook depot", "Quarry Lane store",
                                            "Southbank hub", "Oakmere annex"]),
        record=("transfer", "transfer_id", "T-{n}"), attr=("quantity", "int:5:400", "Number of units to move."),
        tools=dict(find_item=("search_products", "keyword", "Search the product list by name. Returns matching products with their product ids."),
                   find_party=("search_sites", "name", "Search warehouse sites by name. Returns matching sites with their site ids."),
                   list=("list_transfers", "List the open stock transfers going to one site, with transfer ids, product ids, product names and quantities."),
                   create=("create_transfer", "Open a stock transfer of a product to a site."),
                   cancel=("cancel_transfer", "Cancel one open stock transfer."),
                   side=("record_stock_note", "note", "Write a note into the stock log.")),
        ask=dict(create=["open a transfer of {attr} units of {item} to the {party}", "send {attr} of the {item} to the {party}"],
                 cancel=["cancel the transfer of {item} to the {party}", "call off the {item} transfer that is going to the {party}"],
                 move=["change the {item} transfer to the {party} to {new} units", "the {item} transfer to the {party} should be {new} units instead"],
                 side=['write "{text}" into the stock log', 'add a stock log note: "{text}"']),
        notes=["count checked by night shift", "supplier short-shipped", "hold for inspection"], style=1),
    "hotel": dict(
        item=("room", "room_id", "{n}", ["Garden Suite", "Harbour View Double", "Attic Single", "Courtyard Twin", "Lighthouse Suite", "Orchard Family Room",
                                         "Terrace Double", "Library Room"]),
        party=("guest", "guest_id", "G{n}", PEOPLE),
        record=("booking", "booking_id", "BK{n}"), attr=("check_in", "date", "Arrival date, as YYYY-MM-DD."),
        tools=dict(find_item=("lookup_rooms", "room_name", "Find rooms by name. Returns the matching rooms with their numeric room ids."),
                   find_party=("lookup_guests", "guest_name", "Find guest profiles by name. Returns the matching guests with their guest ids."),
                   list=("guest_bookings", "List one guest's bookings, with booking ids, room ids, room names and arrival dates."),
                   create=("book_room", "Book a room for a guest."),
                   cancel=("cancel_booking", "Cancel one booking."),
                   side=("note_for_housekeeping", "message", "Pass a note to housekeeping.")),
        ask=dict(create=["book the {item} for {party}, arriving {attr}", "reserve the {item} for {party} with check-in on {attr}"],
                 cancel=["cancel {party}'s booking of the {item}", "{party} no longer needs the {item}; cancel that booking"],
                 move=["move {party}'s booking of the {item} to a check-in on {new}", "change the arrival date of {party}'s {item} booking to {new}"],
                 side=['tell housekeeping "{text}"', 'send housekeeping this note: "{text}"']),
        notes=["extra pillows requested", "late arrival expected", "feather-free bedding"], style=2),
    "support": dict(
        item=("service", "service_id", "svc-{n}", ["Cloud Backup", "Mail Relay", "Invoice Portal", "Fleet Tracker", "Door Access", "Payroll Export",
                                                   "Status Page", "Video Bridge"]),
        party=("customer", "customer_id", "C-{n}", ["Brightwater Dental", "Kestrel Logistics", "Fennel & Rye", "Atlas Print Works", "Marlowe Clinic",
                                                     "Tidewell Marine", "Copperfield Farms", "Juniper Studio"]),
        record=("ticket", "ticket_id", "TK{n}"), attr=("priority", ["low", "normal", "high", "urgent"], "One of low, normal, high, urgent."),
        tools=dict(find_item=("find_service", "query", "Look up a service by name. Returns matching services with their service ids."),
                   find_party=("find_customer", "query", "Look up a customer account by company name. Returns matching customers with their customer ids."),
                   list=("customer_tickets", "List a customer's open tickets, with ticket ids, service ids, service names and priorities."),
                   create=("open_ticket", "Open a support ticket for a customer about a service."),
                   cancel=("close_ticket", "Close one open ticket."),
                   side=("post_internal_comment", "comment", "Post a comment on the internal support board.")),
        ask=dict(create=["open a {attr} priority ticket for {party} about {item}", "raise a ticket about {item} for {party}, priority {attr}"],
                 cancel=["close {party}'s ticket about {item}", "the {item} ticket for {party} can be closed"],
                 move=["change {party}'s {item} ticket to {new} priority", "re-file {party}'s ticket about {item} as {new} priority"],
                 side=['post "{text}" on the internal board', 'add an internal comment: "{text}"']),
        notes=["customer prefers email", "second report this week", "vendor already contacted"], style=3),
    "shipping": dict(
        item=("route", "route_id", "RT{n}", ["Coastal Express", "Valley Loop", "Northern Trunk", "City Shuttle", "Island Ferry Link", "Overnight Freight",
                                             "Highland Run", "Airport Connector"]),
        party=("recipient", "recipient_id", "rcp_{n}", PEOPLE),
        record=("shipment", "shipment_id", "SH-{n}"), attr=("delivery_date", "date", "Requested delivery date, as YYYY-MM-DD."),
        tools=dict(find_item=("search_routes", "name", "Search delivery routes by name. Returns matching routes with their route ids."),
                   find_party=("search_recipients", "name", "Search the recipient list by name. Returns matching recipients with their recipient ids."),
                   list=("list_shipments", "List the pending shipments for one recipient, with shipment ids, route ids, route names and delivery dates."),
                   create=("schedule_shipment", "Schedule a shipment to a recipient on a route."),
                   cancel=("void_shipment", "Void one pending shipment."),
                   side=("add_dispatch_remark", "remark", "Add a remark to the dispatch sheet.")),
        ask=dict(create=["schedule a shipment to {party} on the {item} for delivery on {attr}", "ship to {party} via the {item}, delivering {attr}"],
                 cancel=["void {party}'s shipment on the {item}", "cancel the {item} shipment going to {party}"],
                 move=["move {party}'s {item} shipment to delivery on {new}", "reschedule the delivery of {party}'s shipment on the {item} to {new}"],
                 side=['add the dispatch remark "{text}"', 'put "{text}" on the dispatch sheet']),
        notes=["fragile, this side up", "gate code needed", "leave with reception"], style=0),
    "courses": dict(
        item=("course", "course_id", "CRS{n}", ["Intro to Ceramics", "Conversational Italian", "Bookkeeping Basics", "Woodturning", "Life Drawing",
                                                 "Beekeeping for Beginners", "Sourdough Baking", "Film Photography"]),
        party=("student", "student_id", "{n}", PEOPLE),
        record=("enrolment", "enrolment_id", "E{n}"), attr=("term", ["spring", "summer", "autumn", "winter"], "One of spring, summer, autumn, winter."),
        tools=dict(find_item=("course_search", "title", "Search evening courses by title. Returns matching courses with their course ids."),
                   find_party=("student_search", "name", "Search students by name. Returns matching students with their numeric student ids."),
                   list=("student_enrolments", "List one student's enrolments, with enrolment ids, course ids, course titles and terms."),
                   create=("enrol_student", "Enrol a student in a course for a term."),
                   cancel=("withdraw_enrolment", "Withdraw one enrolment."),
                   side=("message_registrar", "body", "Send a short message to the registrar's office.")),
        ask=dict(create=["enrol {party} in {item} for the {attr} term", "sign {party} up for {item} in the {attr} term"],
                 cancel=["withdraw {party} from {item}", "take {party} off the {item} course"],
                 move=["move {party}'s {item} enrolment to the {new} term", "switch {party}'s enrolment in {item} to the {new} term"],
                 side=['tell the registrar "{text}"', 'send the registrar this: "{text}"']),
        notes=["fee waiver approved", "needs ground-floor room", "materials already paid"], style=1),
    "vehicles": dict(
        item=("vehicle", "vehicle_id", "V-{n}", ["blue box van", "white minibus", "grey pickup", "red hatchback", "electric cargo bike", "flatbed truck",
                                                 "silver estate car", "green tipper"]),
        party=("driver", "driver_id", "drv{n}", PEOPLE),
        record=("reservation", "reservation_id", "R{n}"), attr=("day", "date", "The day the vehicle is needed, as YYYY-MM-DD."),
        tools=dict(find_item=("find_vehicle", "description", "Find pool vehicles by description. Returns matching vehicles with their vehicle ids."),
                   find_party=("find_driver", "name", "Find registered drivers by name. Returns matching drivers with their driver ids."),
                   list=("driver_reservations", "List one driver's vehicle reservations, with reservation ids, vehicle ids, descriptions and days."),
                   create=("reserve_vehicle", "Reserve a pool vehicle for a driver for one day."),
                   cancel=("release_reservation", "Release one vehicle reservation."),
                   side=("log_fleet_note", "text", "Write a note in the fleet log.")),
        ask=dict(create=["reserve the {item} for {party} on {attr}", "put {party} down for the {item} on {attr}"],
                 cancel=["release {party}'s reservation of the {item}", "{party} does not need the {item} any more; release it"],
                 move=["move {party}'s reservation of the {item} to {new}", "change the day of {party}'s {item} reservation to {new}"],
                 side=['log the fleet note "{text}"', 'write "{text}" in the fleet log']),
        notes=["tyre pressure low", "return with full tank", "tow bar fitted"], style=2),
    "subscriptions": dict(
        item=("plan", "plan_id", "plan_{n}", ["Starter", "Team Plus", "Studio", "Enterprise Flex", "Nonprofit", "Education", "Field Crew", "Archive Only"]),
        party=("account", "account_id", "ACC{n}", ["Halden Architects", "Pinecrest School", "Mosaic Bakery", "Redshore Surveying", "Lumen Theatre",
                                                    "Stoneleigh Vets", "Driftwood Hostel", "Calder Robotics"]),
        record=("subscription", "subscription_id", "sub_{n}"), attr=("seats", "int:1:60", "Number of seats."),
        tools=dict(find_item=("search_plans", "plan_name", "Search plans by name. Returns matching plans with their plan ids."),
                   find_party=("search_accounts", "account_name", "Search accounts by organisation name. Returns matching accounts with their account ids."),
                   list=("account_subscriptions", "List one account's active subscriptions, with subscription ids, plan ids, plan names and seats."),
                   create=("start_subscription", "Start a subscription to a plan for an account."),
                   cancel=("end_subscription", "End one active subscription."),
                   side=("add_billing_memo", "memo", "Attach a memo for the billing team.")),
        ask=dict(create=["start a {item} subscription with {attr} seats for {party}", "set {party} up on the {item} plan with {attr} seats"],
                 cancel=["end {party}'s {item} subscription", "cancel the {item} plan that {party} is on"],
                 move=["change {party}'s {item} subscription to {new} seats", "{party} wants {new} seats on the {item} plan instead"],
                 side=['add the billing memo "{text}"', 'leave billing a memo: "{text}"']),
        notes=["invoice by purchase order", "renewal date unchanged", "discount agreed by phone"], style=3),
    "kitchen": dict(
        item=("dish", "dish_id", "D{n}", ["mushroom risotto", "grilled sea bass", "lamb tagine", "beetroot salad", "pumpkin ravioli", "smoked aubergine dip",
                                          "lemon tart", "braised leeks"]),
        party=("table", "table_id", "tbl-{n}", ["Window Four", "Terrace Two", "Chef's Counter", "Booth Seven", "Garden Six", "Mezzanine One", "Bar End", "Alcove Three"]),
        record=("order", "order_id", "O{n}"), attr=("portions", "int:1:8", "Number of portions."),
        tools=dict(find_item=("menu_search", "dish_name", "Search the menu by dish name. Returns matching dishes with their dish ids."),
                   find_party=("table_search", "table_name", "Search tables by name. Returns matching tables with their table ids."),
                   list=("table_orders", "List the open orders of one table, with order ids, dish ids, dish names and portions."),
                   create=("send_order", "Send an order for a dish to the kitchen for a table."),
                   cancel=("cancel_order", "Cancel one open order."),
                   side=("note_to_pass", "text", "Put a note on the pass for the kitchen staff.")),
        ask=dict(create=["send {attr} portions of the {item} to table {party}", "order the {item}, {attr} portions, for table {party}"],
                 cancel=["cancel the {item} order for table {party}", "table {party} no longer wants the {item}; cancel it"],
                 move=["change table {party}'s {item} order to {new} portions", "table {party} wants {new} portions of the {item} instead"],
                 side=['put "{text}" on the pass', 'leave the kitchen a note: "{text}"']),
        notes=["nut allergy at this table", "serve after the speeches", "birthday candle on dessert"], style=0),
    "tracker": dict(
        item=("project", "project_id", "PRJ-{n}", ["Website Refresh", "Warehouse Scanner App", "Payroll Migration", "Mobile Checkout", "Data Lake Cleanup",
                                                   "Partner Portal", "Onboarding Flow", "Search Rewrite"]),
        party=("developer", "developer_id", "dev_{n}", PEOPLE),
        record=("assignment", "assignment_id", "A-{n}"), attr=("due", "date", "Due date, as YYYY-MM-DD."),
        tools=dict(find_item=("project_lookup", "name", "Look up projects by name. Returns matching projects with their project ids."),
                   find_party=("developer_lookup", "name", "Look up developers by name. Returns matching developers with their developer ids."),
                   list=("developer_assignments", "List one developer's assignments, with assignment ids, project ids, project names and due dates."),
                   create=("assign_project", "Assign a developer to a project with a due date."),
                   cancel=("remove_assignment", "Remove one assignment."),
                   side=("post_standup_note", "note", "Post a note to the stand-up board.")),
        ask=dict(create=["assign {party} to {item}, due {attr}", "put {party} on {item} with a due date of {attr}"],
                 cancel=["remove {party} from {item}", "take {party} off the {item} project"],
                 move=["move the due date of {party}'s {item} assignment to {new}", "{party}'s work on {item} is now due {new}; update it"],
                 side=['post the stand-up note "{text}"', 'put "{text}" on the stand-up board']),
        notes=["blocked on design review", "pairing on Thursday", "scope reduced"], style=1),
    "playlists": dict(
        item=("track", "track_id", "{n}", ["Low Tide", "Copper Wire", "Seven Lanterns", "Northbound", "Paper Planes at Dusk", "Halfway House",
                                           "Static Bloom", "The Long Field"]),
        party=("playlist", "playlist_id", "pl_{n}", ["Morning Run", "Kitchen Radio", "Late Shift", "Road Trip North", "Sunday Slow", "Workshop Mix",
                                                     "Rainy Window", "Garden Party"]),
        record=("entry", "entry_id", "en{n}"), attr=("position", "int:1:40", "Position in the playlist, counting from 1."),
        tools=dict(find_item=("search_tracks", "title", "Search tracks by title. Returns matching tracks with their numeric track ids."),
                   find_party=("search_playlists", "name", "Search playlists by name. Returns matching playlists with their playlist ids."),
                   list=("playlist_entries", "List the entries of one playlist, with entry ids, track ids, track titles and positions."),
                   create=("add_entry", "Add a track to a playlist at a position."),
                   cancel=("remove_entry", "Remove one entry from its playlist."),
                   side=("set_listening_note", "text", "Save a short listening note.")),
        ask=dict(create=['add "{item}" to the {party} playlist at position {attr}', 'put "{item}" at position {attr} of the {party} playlist'],
                 cancel=['remove "{item}" from the {party} playlist', 'take "{item}" off the {party} playlist'],
                 move=['move "{item}" in the {party} playlist to position {new}', 'in the {party} playlist, "{item}" should be at position {new}'],
                 side=['save the listening note "{text}"', 'note down "{text}" as a listening note']),
        notes=["skip the live versions", "good for long drives", "too quiet for the workshop"], style=2),
    "devices": dict(
        item=("device", "device_id", "DV{n}", ["label printer", "barcode scanner", "conference speaker", "drawing tablet", "thermal camera", "portable projector",
                                               "sound level meter", "laser measure"]),
        party=("borrower", "borrower_id", "u{n}", PEOPLE),
        record=("loan", "loan_id", "LN-{n}"), attr=("return_by", "date", "Date the device is due back, as YYYY-MM-DD."),
        tools=dict(find_item=("device_search", "name", "Search the equipment pool by device name. Returns matching devices with their device ids."),
                   find_party=("borrower_search", "name", "Search registered borrowers by name. Returns matching borrowers with their borrower ids."),
                   list=("borrower_loans", "List one borrower's open loans, with loan ids, device ids, device names and due dates."),
                   create=("lend_device", "Lend a device to a borrower until a date."),
                   cancel=("end_loan", "End one open loan."),
                   side=("add_pool_remark", "remark", "Add a remark to the equipment pool log.")),
        ask=dict(create=["lend the {item} to {party} until {attr}", "sign the {item} out to {party}, due back {attr}"],
                 cancel=["end {party}'s loan of the {item}", "{party} has brought the {item} back; end that loan"],
                 move=["extend {party}'s loan of the {item} to {new}", "change the due date of {party}'s {item} loan to {new}"],
                 side=['add the pool remark "{text}"', 'write "{text}" in the equipment pool log']),
        notes=["charger missing", "firmware updated", "case is cracked"], style=3),
    # ---- held out: never in the training file; run live by bench/recovery_eval.py ----
    "tool_hire": dict(
        heldout=True,
        item=("machine", "machine_id", "MX{n}", ["floor sander", "tile cutter", "pressure washer", "cement mixer", "hedge trimmer", "wallpaper steamer",
                                                 "post hole borer", "carpet cleaner"]),
        party=("hirer", "hirer_id", "hr_{n}", PEOPLE),
        record=("hire", "hire_id", "HI{n}"), attr=("collect_on", "date", "Collection date, as YYYY-MM-DD."),
        tools=dict(find_item=("find_machines", "name", "Search hire machines by name. Returns matching machines with their machine ids."),
                   find_party=("find_hirers", "name", "Search hire customers by name. Returns matching hirers with their hirer ids."),
                   list=("hirer_hires", "List one hirer's upcoming hires, with hire ids, machine ids, machine names and collection dates."),
                   create=("create_hire", "Create a hire of a machine for a hirer."),
                   cancel=("cancel_hire", "Cancel one upcoming hire."),
                   side=("yard_note", "text", "Leave a note for the yard staff.")),
        ask=dict(create=["hire out the {item} to {party}, collecting on {attr}", "set up a hire of the {item} for {party} with collection on {attr}"],
                 cancel=["cancel {party}'s hire of the {item}", "{party} will not need the {item}; cancel the hire"],
                 move=["move {party}'s hire of the {item} to collection on {new}", "change the collection date of {party}'s {item} hire to {new}"],
                 side=['leave the yard a note saying "{text}"', 'tell the yard staff "{text}"']),
        notes=["needs a trailer", "deposit paid in cash", "show how to change the blade"], style=1),
    "parking": dict(
        heldout=True,
        item=("car park", "carpark_id", "{n}", ["Mill Street", "Quayside", "Old Brewery Yard", "Station Approach", "Market Square", "Canal Wharf",
                                                "Hilltop", "Abbey Gate"]),
        party=("resident", "resident_id", "RS-{n}", PEOPLE),
        record=("permit", "permit_id", "P{n}"), attr=("months", "int:1:12", "Length of the permit in months."),
        tools=dict(find_item=("carpark_lookup", "name", "Look up car parks by name. Returns matching car parks with their numeric car park ids."),
                   find_party=("resident_lookup", "name", "Look up residents by name. Returns matching residents with their resident ids."),
                   list=("resident_permits", "List one resident's active permits, with permit ids, car park ids, car park names and lengths in months."),
                   create=("issue_permit", "Issue a parking permit for a resident at a car park."),
                   cancel=("revoke_permit", "Revoke one active permit."),
                   side=("warden_notice", "notice", "Send a notice to the parking wardens.")),
        ask=dict(create=["issue {party} a {attr}-month permit for the {item} car park", "give {party} a permit at {item} for {attr} months"],
                 cancel=["revoke {party}'s permit for the {item} car park", "{party} has moved away; revoke the {item} permit"],
                 move=["change {party}'s {item} permit to {new} months", "{party}'s permit at {item} should run for {new} months instead"],
                 side=['send the wardens the notice "{text}"', 'notify the wardens: "{text}"']),
        notes=["blue badge holder", "second vehicle on file", "bay 12 is out of use"], style=3),
}

WRAPPERS = ["{}.", "Please {}.", "Can you {}?", "I need you to {}.", "{} please.", "Could you {}?"]
JOINERS = [", then ", " and ", ", and after that ", ", and also "]
PATTERNS = {"lookup_create": 20, "lookup_cancel": 15, "recover_create": 14, "recover_cancel": 11, "move": 15, "partial": 10, "ask": 10, "direct": 5}
TRAIN_TOPICS = [name for name, topic in TOPICS.items() if not topic.get("heldout")]
HELDOUT_TOPICS = [name for name, topic in TOPICS.items() if topic.get("heldout")]


def all_tool_names() -> set[str]:
    return {spec[0] for topic in TOPICS.values() for spec in topic["tools"].values()}


class World:
    """One topic's data for one conversation, and the tools that act on it."""

    def __init__(self, topic_name: str, rng: random.Random):
        self.name, self.t, self.rng = topic_name, TOPICS[topic_name], rng
        t = self.t
        self.numeric = {kind: "{n}" == t[kind][2] for kind in ("item", "party")}
        self.items = self._named("item", 6)
        self.parties = self._named("party", 7)
        self.records: dict = {}
        self.side_log: list[str] = []
        self._next = rng.randrange(100, 900)
        pairs = rng.sample([(i, p) for i in self.items for p in self.parties], rng.randrange(3, 6))
        for item, party in pairs:                                # no two records join the same item and party
            self._add(item, party, self.attr_value())

    def _named(self, kind: str, count: int) -> dict:
        noun, _, pattern, names = self.t[kind]
        if names == PEOPLE:
            firsts, lasts = self.rng.sample(FIRST, count), self.rng.sample(LAST, count)
            names = [f"{a} {b}" for a, b in zip(firsts, lasts)]
        else:
            names = self.rng.sample(names, min(count, len(names)))
        numbers = self.rng.sample(range(1000, 9999) if not self.numeric[kind] else range(10000, 99999), len(names))
        return {(n if self.numeric[kind] else pattern.format(n=n)): name for n, name in zip(numbers, names)}

    def attr_value(self, avoid=None):
        kind = self.t["attr"][1]
        while True:
            if kind == "date":
                value = f"{self.rng.choice([2025, 2026, 2027])}-{self.rng.randrange(1, 13):02d}-{self.rng.randrange(1, 29):02d}"
            elif isinstance(kind, list):
                value = self.rng.choice(kind)
            else:
                _, low, high = kind.split(":")
                value = self.rng.randrange(int(low), int(high) + 1)
            if value != avoid:
                return value

    def _add(self, item_id, party_id, attr):
        self._next += self.rng.randrange(1, 40)
        record_id = self.t["record"][2].format(n=self._next)
        self.records[record_id] = {"item": item_id, "party": party_id, "attr": attr}
        return record_id

    def state(self) -> dict:
        return {"records": sorted((r["item"], r["party"], r["attr"]) for r in self.records.values()), "side": list(self.side_log)}

    # ---- tool schemas ----
    def tools(self) -> list[dict]:
        t, tools = self.t, self.t["tools"]
        (item_noun, item_key, _, _), (party_noun, party_key, _, _) = t["item"], t["party"]
        record_noun, record_key, _ = t["record"]
        attr_name, attr_kind, attr_description = t["attr"]
        attr_schema = ({"type": "string", "enum": attr_kind} if isinstance(attr_kind, list) else
                       {"type": "string"} if attr_kind == "date" else {"type": "integer"})

        def id_schema(kind: str, noun: str, source: str) -> dict:
            return {"type": "integer" if self.numeric[kind] else "string", "description": f"Id of the {noun}, as returned by {source}."}

        def tool(name: str, description: str, properties: dict) -> dict:
            return {"type": "function", "function": {"name": name, "description": description, "parameters": {
                "type": "object", "properties": properties, "required": list(properties)}}}

        return [
            tool(tools["find_item"][0], tools["find_item"][2], {tools["find_item"][1]: {"type": "string", "description": f"Name or part of the name of the {item_noun}."}}),
            tool(tools["find_party"][0], tools["find_party"][2], {tools["find_party"][1]: {"type": "string", "description": f"Name or part of the name of the {party_noun}."}}),
            tool(tools["list"][0], tools["list"][1], {party_key: id_schema("party", party_noun, tools["find_party"][0])}),
            tool(tools["create"][0], tools["create"][1], {item_key: id_schema("item", item_noun, tools["find_item"][0]),
                                                          party_key: id_schema("party", party_noun, tools["find_party"][0]),
                                                          attr_name: {**attr_schema, "description": attr_description}}),
            tool(tools["cancel"][0], tools["cancel"][1], {record_key: {"type": "string", "description": f"Id of the {record_noun}, as returned by {tools['list'][0]} or {tools['create'][0]}."}}),
            tool(tools["side"][0], tools["side"][2], {tools["side"][1]: {"type": "string", "description": "The text."}}),
        ]

    # ---- tool behaviour ----
    def _error(self, noun: str, value) -> str:
        style = self.t["style"]
        if style == 0:
            return json.dumps({"error": f"{noun} '{value}' does not exist"}, ensure_ascii=False)
        if style == 1:
            return json.dumps({"status": "failed", "reason": f"unknown {noun} id: {value}"}, ensure_ascii=False)
        if style == 2:
            return json.dumps({"success": False, "message": f"{value} is not a valid {noun} id"}, ensure_ascii=False)
        return f"Error: no such {noun} ({value})"

    def _list(self, rows: list[dict]) -> str:
        style = self.t["style"]
        if style == 0:
            return json.dumps(rows, ensure_ascii=False)
        if style == 1:
            return json.dumps({"results": rows}, ensure_ascii=False)
        if style == 2:
            return json.dumps({"matches": rows, "count": len(rows)}, ensure_ascii=False)
        return json.dumps({"found": len(rows), "items": rows}, ensure_ascii=False)

    def _ok(self, **fields) -> str:
        style = self.t["style"]
        head = [{"ok": True}, {"status": "done"}, {"success": True}, {"result": "ok"}][style]
        return json.dumps({**head, **fields}, ensure_ascii=False)

    def run(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Execute one call. Returns the tool's reply and whether it was an error."""
        t, tools = self.t, self.t["tools"]
        (item_noun, item_key, _, _), (party_noun, party_key, _, _) = t["item"], t["party"]
        record_noun, record_key, _ = t["record"]
        attr_name = t["attr"][0]
        if not isinstance(arguments, dict):
            return json.dumps({"error": "arguments must be an object"}), True
        for kind, table, noun, key in (("find_item", self.items, item_noun, item_key), ("find_party", self.parties, party_noun, party_key)):
            if name == tools[kind][0]:
                query = str(arguments.get(tools[kind][1], "")).strip().lower()
                if not query:
                    return json.dumps({"error": f"missing parameter {tools[kind][1]}"}), True
                return self._list([{key: id_, "name": text} for id_, text in table.items() if query in text.lower()]), False
        if name == tools["list"][0]:
            party = arguments.get(party_key)
            if party not in self.parties:
                return self._error(party_noun, party), True
            return self._list([{record_key: rid, item_key: r["item"], "name": self.items[r["item"]], attr_name: r["attr"]}
                               for rid, r in self.records.items() if r["party"] == party]), False
        if name == tools["create"][0]:
            item, party = arguments.get(item_key), arguments.get(party_key)
            if item not in self.items:
                return self._error(item_noun, item), True
            if party not in self.parties:
                return self._error(party_noun, party), True
            if attr_name not in arguments:
                return json.dumps({"error": f"missing parameter {attr_name}"}), True
            return self._ok(**{record_key: self._add(item, party, arguments[attr_name])}), False
        if name == tools["cancel"][0]:
            record = arguments.get(record_key)
            if record not in self.records:
                return self._error(record_noun, record), True
            del self.records[record]
            return self._ok(), False
        if name == tools["side"][0]:
            text = arguments.get(tools["side"][1])
            if not isinstance(text, str) or not text:
                return json.dumps({"error": f"missing parameter {tools['side'][1]}"}), True
            self.side_log.append(text)
            return self._ok(), False
        return json.dumps({"error": f"unknown tool {name}"}), True


def sentence(rng: random.Random, *parts: str) -> str:
    text = parts[0]
    for part in parts[1:]:
        text += rng.choice(JOINERS) + part
    # "Please ..." and "Can you ...?" only fit a request that starts with a verb
    statement = (text[0].isupper() or text.startswith(("the ", "table ", "in ", '"')) or ";" in parts[0]
                 or any(mark in parts[0] for mark in (" instead", " should be ", " is now ", " can be ")))
    text = text + "." if statement else rng.choice(WRAPPERS).format(text)
    return text[0].upper() + text[1:]


class Script:
    """Builds one conversation: the user's request, the calls a careful assistant makes, the
    tool replies, and the assistant's last message."""

    def __init__(self, world: World, rng: random.Random):
        self.w, self.rng, self.messages, self.context_only = world, rng, [], set()

    def call(self, kind: str, arguments: dict, context_only: bool = False) -> str:
        name = self.w.t["tools"][kind][0]
        if context_only:
            self.context_only.add(len(self.messages))
        self.messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]})
        reply, _ = self.w.run(name, arguments)
        self.messages.append({"role": "tool", "content": reply})
        return reply

    def find(self, kind: str, name: str) -> list:
        tools = self.w.t["tools"]
        self.call(kind, {tools[kind][1]: name})
        table = self.w.items if kind == "find_item" else self.w.parties
        return [id_ for id_, text in table.items() if name.lower() in text.lower()]

    def say(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})


def build(topic_name: str, pattern: str, seed: int) -> dict:
    """One conversation. Returns the tool list, the messages, which assistant turns are
    context only, the world's state before and after (what the live check compares), whether
    the conversation should end with a question to the user, and the world as it was when the
    user asked (for a live run)."""
    rng = random.Random(seed)
    w = World(topic_name, rng)
    t, tools, s = w.t, w.t["tools"], Script(w, rng)
    (item_noun, item_key, _, _), (party_noun, party_key, _, _) = t["item"], t["party"]
    record_noun, record_key, _ = t["record"]
    attr_name = t["attr"][0]
    before, initial = w.state(), copy.deepcopy(w)
    ask = lambda kind, **values: rng.choice(t["ask"][kind]).format(**values)   # noqa: E731

    def new_pair():
        """An item and a party that have no record together yet."""
        while True:
            item, party = rng.choice(list(w.items)), rng.choice(list(w.parties))
            if not any(r["item"] == item and r["party"] == party for r in w.records.values()):
                return item, party

    def existing():
        record_id = rng.choice(list(w.records))
        return record_id, w.records[record_id]

    def unique(table: dict, id_) -> str:
        """The name to say for this id: the full name, which matches only this entry."""
        return table[id_]

    ends_with_question = False
    if pattern in ("lookup_create", "recover_create"):
        item, party = new_pair()
        value = w.attr_value()
        s.messages.append({"role": "user", "content": sentence(rng, ask("create", item=unique(w.items, item), party=unique(w.parties, party), attr=value))})
        if pattern == "recover_create":
            wrong = {item_key: w.items[item], party_key: w.parties[party], attr_name: value}
            if rng.random() < 0.3:                                   # only one of the two ids was made up
                wrong[item_key] = item
                s.messages[0]["content"] += f" The {item_noun} id is {item}."
            s.call("create", wrong, context_only=True)
            if wrong[item_key] != item:
                s.find("find_item", w.items[item])
        else:
            s.find("find_item", w.items[item])
        s.find("find_party", w.parties[party])
        reply = json.loads(s.call("create", {item_key: item, party_key: party, attr_name: value}))
        record_id = reply[record_key]
        s.say(rng.choice([
            f"Done. The {record_noun} is created with id {record_id}: {w.items[item]} for {w.parties[party]}, {attr_name.replace('_', ' ')} {value}.",
            f"The {record_noun} for {w.parties[party]} ({w.items[item]}, {attr_name.replace('_', ' ')} {value}) is in place. Its id is {record_id}.",
            f"All set: {w.items[item]} for {w.parties[party]}, {attr_name.replace('_', ' ')} {value}. The {record_noun} id is {record_id}."]))
    elif pattern in ("lookup_cancel", "recover_cancel"):
        record_id, record = existing()
        item, party = record["item"], record["party"]
        s.messages.append({"role": "user", "content": sentence(rng, ask("cancel", item=w.items[item], party=w.parties[party]))})
        if pattern == "recover_cancel":
            made_up = rng.choice([f"{w.items[item]} for {w.parties[party]}", w.items[item], f"{w.parties[party]} {w.items[item]}"])
            s.call("cancel", {record_key: made_up}, context_only=True)
        s.find("find_party", w.parties[party])
        s.call("list", {party_key: party})
        target = next(rid for rid, r in w.records.items() if r["party"] == party and r["item"] == item)
        s.call("cancel", {record_key: target})
        s.say(rng.choice([
            f"Done. {w.parties[party]}'s {record_noun} for {w.items[item]} ({target}) is cancelled.",
            f"The {record_noun} {target} ({w.items[item]}, {w.parties[party]}) has been removed.",
            f"That {record_noun} is gone: {target}, {w.items[item]} for {w.parties[party]}."]))
    elif pattern == "move":
        record_id, record = existing()
        item, party, old = record["item"], record["party"], record["attr"]
        value = w.attr_value(avoid=old)
        s.messages.append({"role": "user", "content": sentence(rng, ask("move", item=w.items[item], party=w.parties[party], new=value))})
        if rng.random() < 0.3:
            s.call("cancel", {record_key: f"{w.items[item]} for {w.parties[party]}"}, context_only=True)
        s.find("find_party", w.parties[party])
        s.call("list", {party_key: party})
        target = next(rid for rid, r in w.records.items() if r["party"] == party and r["item"] == item)
        new_id = json.loads(s.call("create", {item_key: item, party_key: party, attr_name: value}))[record_key]
        s.call("cancel", {record_key: target})
        label = attr_name.replace("_", " ")
        s.say(rng.choice([
            f"Done. I created a new {record_noun} ({new_id}) for {w.items[item]} and {w.parties[party]} with {label} {value}, and cancelled the old one ({target}, {label} {old}).",
            f"{w.parties[party]}'s {record_noun} for {w.items[item]} now has {label} {value}. The new id is {new_id}; the old {record_noun} {target} is cancelled.",
            f"Moved: the new {record_noun} {new_id} has {label} {value}, and the old one ({target}) is removed."]))
    elif pattern == "partial":
        record_id, record = existing()
        item, party = record["item"], record["party"]
        first = ask("cancel", item=w.items[item], party=w.parties[party])
        if rng.random() < 0.5:
            # second part: a create for somebody who is not in the system
            ghost = f"{rng.choice(FIRST)} {rng.choice(LAST)}" if t["party"][3] == PEOPLE else rng.choice(["Elmwood", "Fairhaven", "Greystone", "Larkfield"]) + " " + rng.choice(["Nine", "Annex", "Works", "North"])
            while any(ghost.lower() in name.lower() for name in w.parties.values()):
                ghost += "son"
            other_item, value = rng.choice(list(w.items)), w.attr_value()
            second = ask("create", item=w.items[other_item], party=ghost, attr=value)
            s.messages.append({"role": "user", "content": sentence(rng, first, second)})
            s.find("find_party", w.parties[party])
            s.call("list", {party_key: party})
            target = next(rid for rid, r in w.records.items() if r["party"] == party and r["item"] == item)
            s.call("cancel", {record_key: target})
            s.find("find_item", w.items[other_item])
            s.find("find_party", ghost)
            s.say(rng.choice([
                f"I cancelled {w.parties[party]}'s {record_noun} for {w.items[item]} ({target}). I could not do the second part: there is no {party_noun} named {ghost}, so nothing was created for {w.items[other_item]}.",
                f"One of the two is done. {record_noun.capitalize()} {target} ({w.items[item]}, {w.parties[party]}) is cancelled. The {record_noun} for {ghost} was not created, because the search for that {party_noun} found nobody by that name.",
                f"Cancelled: {target}, {w.items[item]} for {w.parties[party]}. Not done: the new {record_noun} for {ghost}. No {party_noun} with that name exists, so I need the right name before I can create it."]))
        else:
            # first part cannot be done: the party has no such record; the side note still goes through
            other = next(i for i in w.items if not any(r["item"] == i and r["party"] == party for r in w.records.values()))
            note = rng.choice(t["notes"])
            s.messages.append({"role": "user", "content": sentence(rng, ask("cancel", item=w.items[other], party=w.parties[party]), ask("side", text=note))})
            s.find("find_party", w.parties[party])
            s.call("list", {party_key: party})
            s.call("side", {tools["side"][1]: note})
            s.say(rng.choice([
                f"The note is saved. I could not cancel anything: {w.parties[party]} has no {record_noun} for {w.items[other]}.",
                f"Only the second part is done: the note \"{note}\" is recorded. There is no {record_noun} for {w.items[other]} under {w.parties[party]}, so there was nothing to cancel.",
                f"I recorded the note, but the cancellation was not possible. {w.parties[party]}'s list has no {record_noun} for {w.items[other]}."]))
    elif pattern == "ask":
        item = rng.choice(list(w.items))
        value = w.attr_value()
        ends_with_question = True
        if rng.random() < 0.5 and t["party"][3] == PEOPLE:
            # two people share the first name the user gives
            ids = list(w.parties)
            first_name = w.parties[ids[0]].split()[0]
            w.parties[ids[1]] = f"{first_name} {w.parties[ids[1]].split()[1]}"
            before, initial = w.state(), copy.deepcopy(w)
            s.messages.append({"role": "user", "content": sentence(rng, ask("create", item=w.items[item], party=first_name, attr=value))})
            s.find("find_item", w.items[item])
            found = s.find("find_party", first_name)
            listed = " and ".join(f"{w.parties[i]} ({i})" for i in found)
            s.say(rng.choice([
                f"There are {len(found)} {party_noun}s called {first_name}: {listed}. Which one do you mean?",
                f"I found more than one {party_noun} named {first_name}: {listed}. Which of them is this for?"]))
        else:
            ghost = f"{rng.choice(FIRST)} {rng.choice(LAST)}" if t["party"][3] == PEOPLE else rng.choice(["Elmwood", "Fairhaven", "Greystone", "Larkfield"]) + " " + rng.choice(["Nine", "Annex", "Works", "North"])
            while any(ghost.lower() in name.lower() for name in w.parties.values()):
                ghost += "son"
            s.messages.append({"role": "user", "content": sentence(rng, ask("create", item=w.items[item], party=ghost, attr=value))})
            s.find("find_item", w.items[item])
            s.find("find_party", ghost)
            s.say(rng.choice([
                f"I could not find a {party_noun} named {ghost}. Could you check the name, or tell me who you mean?",
                f"The search found no {party_noun} called {ghost}, so I have not created anything. What is the exact name?"]))
    elif pattern == "direct":
        if rng.random() < 0.5:
            record_id, record = existing()
            s.messages.append({"role": "user", "content": sentence(rng, f"cancel {record_noun} {record_id}")})
            s.call("cancel", {record_key: record_id})
            s.say(rng.choice([f"Done. {record_noun.capitalize()} {record_id} is cancelled.", f"{record_noun.capitalize()} {record_id} has been removed."]))
        else:
            item, party = new_pair()
            value = w.attr_value()
            s.messages.append({"role": "user", "content": sentence(rng, f"use {tools['create'][0]} with {item_noun} id {item}, {party_noun} id {party} and {attr_name.replace('_', ' ')} {value}")})
            record_id = json.loads(s.call("create", {item_key: item, party_key: party, attr_name: value}))[record_key]
            s.say(rng.choice([f"Done. The new {record_noun} has id {record_id}.", f"Created {record_noun} {record_id}."]))
    else:
        raise ValueError(pattern)
    return {"topic": topic_name, "pattern": pattern, "seed": seed, "tools": w.tools(), "messages": s.messages,
            "context_only": sorted(s.context_only), "state_before": before, "state_after": w.state(),
            "ends_with_question": ends_with_question, "world": initial}


def conversations(topics: list[str], count: int, seed: int) -> list[dict]:
    """`count` conversations over the topics, patterns in the shares of PATTERNS."""
    rng = random.Random(seed)
    names, weights = list(PATTERNS), list(PATTERNS.values())
    out = []
    for index in range(count):
        out.append(build(topics[index % len(topics)], rng.choices(names, weights)[0], seed * 1_000_003 + index))
    return out
