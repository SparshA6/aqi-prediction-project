from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import csv
import glob
from pathlib import Path
import os
import sys
import time

from paths import CITIES_CSV, ROOT, STATION_META_DIR, STATIONS_CSV


# ============================================================
# CPCB CCR portal downloader -- hourly station AQI, many stations / cities
#
#   python src/cpcb_downloader.py [--auto] [--city Lucknow Jaipur ...]
#
# Same portal flow as legacy/cpcb_downloader_single_station.py (State -> City ->
# Station -> Year -> Submit -> one download per month), looped over every city in
# config/cities.csv and every station of that city in config/stations.csv, all in
# ONE browser session so the CAPTCHA is only solved once.
#
#   * Stations whose folder already has a full year are skipped -> safely
#     re-runnable / resumable (Ctrl+C any time, rerun to continue).
#   * A failing station (not in the portal dropdown, no data, ...) is reported at
#     the end and doesn't stop the rest. If a whole city's State/City can't be
#     selected, its stations fail and the next city is tried.
#   * For each city the script saves the portal's own station list to
#     data/raw/station_metadata/portal_<City>.txt and prints stations the portal
#     offers that are NOT in stations.csv, and stations.csv entries the portal
#     doesn't offer under that exact name (agency labels differ between sources:
#     e.g. Delhi's "- IMD" sites are "- IITM" on the portal).
#   * --auto: don't wait for a terminal ENTER; wait for the CAPTCHA modal to close
#     after you solve it in the browser window. Lets it run in the background;
#     the CAPTCHA itself is still solved by a human.
# ============================================================

AUTO = "--auto" in sys.argv

YEAR = "2023"
if "--year" in sys.argv:
    i = sys.argv.index("--year")
    if i + 1 == len(sys.argv) or not sys.argv[i + 1].isdigit():
        raise SystemExit("--year requires a four-digit year")
    YEAR = sys.argv[i + 1]

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

if '--months' in sys.argv:
    i=sys.argv.index('--months')+1
    requested=[]
    while i<len(sys.argv) and not sys.argv[i].startswith('--'):
        try: requested.append(int(sys.argv[i]))
        except ValueError: raise SystemExit('--months requires integers 1 through 12')
        i+=1
    if not requested or len(set(requested))!=len(requested) or any(m<1 or m>12 for m in requested):
        raise SystemExit('--months requires distinct integers 1 through 12')
    MONTHS=[MONTHS[m-1] for m in sorted(requested)]


def selected_city_names():
    if "--city" in sys.argv:
        i = sys.argv.index("--city") + 1
        names = []
        while i < len(sys.argv) and not sys.argv[i].startswith("--"):
            names.append(sys.argv[i])
            i += 1
        return names
    return None


def load_cities():
    with open(CITIES_CSV, newline="", encoding="utf-8") as f:
        cities = list(csv.DictReader(f))
    wanted = selected_city_names()
    if wanted:
        unknown = [w for w in wanted if w not in {c["city"] for c in cities}]
        if unknown:
            raise SystemExit(f"Unknown city/cities {unknown}. Known: {[c['city'] for c in cities]}")
        cities = [c for c in cities if c["city"] in wanted]
    return cities


def load_stations(city):
    with open(STATIONS_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["city"].strip() == city]
    output_root = None
    if "--output-root" in sys.argv:
        i = sys.argv.index("--output-root")
        if i + 1 == len(sys.argv):
            raise SystemExit("--output-root requires a directory")
        output_root = Path(sys.argv[i + 1]).resolve()
    stations = []
    for r in rows:
        folder = ROOT / r["data_dir"]
        if output_root is not None:
            relative = Path(r["data_dir"]).relative_to(Path("data/raw/cpcb"))
            folder = output_root / relative
        stations.append({"name": r["station_name"], "folder": str(folder)})
    return stations


def is_complete(folder):
    """True if the folder already has one file per month for YEAR."""
    files = glob.glob(os.path.join(folder, "*.xlsx"))
    months_found = {m for m in MONTHS for f in files if m.lower() in os.path.basename(f).lower() and YEAR in os.path.basename(f)}
    return len(months_found) == len(MONTHS)


# ============================================================
# CAPTCHA HANDLING
# ============================================================

def wait_for_captcha(page):
    """
    Wait for the user to manually solve CPCB CAPTCHA.
    """

    print()
    print("=" * 65)
    print("                 CPCB CAPTCHA")
    print("=" * 65)
    print()
    print("If a CAPTCHA is visible in the browser:")
    print()
    print("  1. Solve the CAPTCHA manually.")
    print("  2. Wait until the CAPTCHA disappears.")
    print("  3. Press ENTER in this terminal.")
    print()
    print("=" * 65)

    if AUTO:
        print("Auto mode: solve the CAPTCHA in the browser window; continuing when it closes (waits up to 30 min)...")
        try:
            page.locator("#captchaModal").wait_for(state="visible", timeout=20000)
        except PlaywrightTimeoutError:
            print("No CAPTCHA appeared -- continuing.")
        else:
            page.locator("#captchaModal").wait_for(state="hidden", timeout=1800000)
            print("âœ“ CAPTCHA solved.")
    else:
        input("Press ENTER after solving the CAPTCHA... ")

    # Give Angular some time to remove the overlay
    page.wait_for_timeout(2000)

    # Wait until CAPTCHA modal disappears
    try:
        page.locator("#captchaModal").wait_for(
            state="hidden",
            timeout=10000
        )
        print("âœ“ CAPTCHA modal closed.")
    except PlaywrightTimeoutError:
        print("WARNING: CAPTCHA modal may still be present.")
        print("Please make sure it is closed before continuing.")
        if not AUTO:
            input("Press ENTER to continue anyway... ")

    # Wait for loading overlay to disappear
    try:
        page.locator("#loader-overlay").wait_for(
            state="hidden",
            timeout=15000
        )
        print("âœ“ CPCB loading overlay removed.")
    except PlaywrightTimeoutError:
        print("WARNING: Loader overlay is still present.")


# ============================================================
# WAIT UNTIL PAGE IS READY
# ============================================================

def wait_until_ready(page):

    print("Waiting for CPCB page to become ready...")

    # CAPTCHA -- can reappear between stations on some sessions
    try:
        captcha = page.locator("#captchaModal")

        if captcha.is_visible(timeout=3000):
            wait_for_captcha(page)

    except:
        pass

    # Loader
    try:
        page.locator("#loader-overlay").wait_for(
            state="hidden",
            timeout=20000
        )
    except:
        pass

    page.wait_for_timeout(1000)

    print("âœ“ Page ready.")


# ============================================================
# SELECT ANGULAR DROPDOWN
# ============================================================

def select_dropdown(page, dropdown_number, value, name):

    print(f"Selecting {name}: {value}")

    # Wait for overlays to disappear
    wait_until_ready(page)

    # CPCB uses Angular dropdown components.
    # The order is:
    #
    # 0 = Type
    # 1 = State
    # 2 = City
    # 3 = Station
    # 4 = Year

    dropdowns = page.locator("ng-select")

    count = dropdowns.count()

    print(f"Found {count} dropdowns.")

    if count <= dropdown_number:
        raise Exception(
            f"Could not find dropdown #{dropdown_number} "
            f"for {name}"
        )

    dropdown = dropdowns.nth(dropdown_number)

    # Click actual dropdown
    dropdown.click()

    page.wait_for_timeout(500)

    # Angular ng-select normally creates .ng-option
    option = page.locator(
        ".ng-option"
    ).filter(
        has_text=value
    )

    if option.count() == 0:

        print()
        print("Available options found:")

        options = page.locator(".ng-option")

        for i in range(min(options.count(), 80)):
            try:
                print(
                    "   ",
                    options.nth(i).inner_text()
                )
            except:
                pass

        page.keyboard.press("Escape")

        raise Exception(
            f"Could not find option '{value}' "
            f"for {name}"
        )

    # Several stations share a name prefix (e.g. "Pusa, Delhi - DPCC" vs
    # "Pusa, Delhi - IMD"), so prefer an exact text match over the first
    # substring match.
    exact = page.locator(".ng-option").filter(
        has_text=value
    )
    target = None
    for i in range(exact.count()):
        try:
            if exact.nth(i).inner_text().strip() == value:
                target = exact.nth(i)
                break
        except:
            continue

    if target is None:
        # A substring hit can be another site ("IMD Lodhi Road" for "Lodhi Road"),
        # whose data would then be saved under this station's folder.
        if name == "Station":
            page.keyboard.press("Escape")
            raise Exception(f"No exact option '{value}' for {name}")
        target = exact.first

    target.click()

    page.wait_for_timeout(1000)

    print(f"âœ“ {name} selected.")


# ============================================================
# LIST WHAT THE PORTAL OFFERS
# ============================================================

def list_portal_stations(page, state, city):
    """Open the Station dropdown for state/city and return every option's text."""

    select_dropdown(page, 1, state, "State")
    select_dropdown(page, 2, city, "City")

    wait_until_ready(page)

    page.locator("ng-select").nth(3).click()
    page.wait_for_timeout(1000)

    options = page.locator(".ng-option").all_inner_texts()
    options = [o.strip() for o in options if o.strip()]

    page.keyboard.press("Escape")
    page.wait_for_timeout(500)

    return options


# ============================================================
# CLICK SUBMIT
# ============================================================

def click_submit(page):

    print("Clicking Submit...")

    wait_until_ready(page)

    submit = page.get_by_role(
        "button",
        name="Submit"
    )

    submit.click()

    print("âœ“ Submit clicked.")

    # Wait for results
    page.wait_for_timeout(3000)

    wait_until_ready(page)


# ============================================================
# DOWNLOAD MONTH
# ============================================================

def download_month(page, month, download_folder):

    print()
    print("-" * 50)
    print(f"Downloading: {month}")
    print("-" * 50)

    # Find table row containing month
    rows = page.locator("tr")

    matching_row = None

    for i in range(rows.count()):

        row = rows.nth(i)

        try:
            text = row.inner_text()

            if month.lower() in text.lower():
                matching_row = row
                break

        except:
            continue

    if matching_row is None:

        print(f"âœ— Could not find row for {month}")

        return False

    # Find button in row
    buttons = matching_row.locator("button")

    if buttons.count() == 0:

        print("No button found, checking links...")

        buttons = matching_row.locator("a")

    if buttons.count() == 0:

        print(f"âœ— No download button for {month}")

        return False

    button = buttons.last

    try:

        with page.expect_download(
            timeout=30000
        ) as download_info:

            button.click()

        download = download_info.value

        filename = download.suggested_filename

        save_path = os.path.join(
            download_folder,
            filename
        )

        download.save_as(save_path)

        print(f"âœ“ Downloaded: {filename}")

        return True

    except Exception as e:

        print(f"âœ— Download failed:")
        print(e)

        return False


# ============================================================
# DOWNLOAD ONE STATION (state/city/station/year -> submit -> all months)
# ============================================================

def download_station(page, state, city, station_name, download_folder):

    print()
    print("=" * 65)
    print(f"STATION: {station_name}   [{city}]")
    print("=" * 65)

    os.makedirs(download_folder, exist_ok=True)

    select_dropdown(page, 1, state, "State")
    select_dropdown(page, 2, city, "City")
    select_dropdown(page, 3, station_name, "Station")
    select_dropdown(page, 4, YEAR, "Year")

    click_submit(page)

    successful = 0

    for month in MONTHS:

        result = download_month(page, month, download_folder)

        if result:
            successful += 1

        time.sleep(2)

    print()
    print(f"{station_name}: {successful}/{len(MONTHS)} months downloaded -> {download_folder}")

    return successful


# ============================================================
# MAIN
# ============================================================

cities = load_cities()

plan = []   # (city_row, all_stations, todo_stations)
for c in cities:
    stations = load_stations(c["city"])
    todo = [s for s in stations if not is_complete(s["folder"])]
    print(f"{c['city']:<10} {len(stations):>3} stations in stations.csv, {len(stations) - len(todo):>3} complete, {len(todo):>3} to download")
    if todo:
        plan.append((c, stations, todo))

if not plan:
    print("Nothing to do.")
    raise SystemExit

print(f"\n{sum(len(t) for _, _, t in plan)} stations to download across {len(plan)} cities.")

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    context = browser.new_context(
        accept_downloads=True
    )

    page = context.new_page()

    page.set_default_timeout(15000)

    print()
    print("Opening CPCB AQI Repository...")

    page.goto(
        "https://airquality.cpcb.gov.in/ccr/#/repository/aqi",
        wait_until="domcontentloaded"
    )

    print("Page loaded.")

    # CAPTCHA is solved once, here, before the loops.
    wait_for_captcha(page)

    results = {}   # (city, station) -> months downloaded

    for c, stations, todo in plan:

        city, state, portal_city = c["city"], c["portal_state"], c["portal_city"]

        print()
        print("#" * 65)
        print(f"# CITY: {city}  (portal: {state} / {portal_city})")
        print("#" * 65)

        # ---- what does the portal offer for this city? ----
        try:
            portal_options = list_portal_stations(page, state, portal_city)
            STATION_META_DIR.mkdir(parents=True, exist_ok=True)
            (STATION_META_DIR / f"portal_{city}.txt").write_text("\n".join(portal_options), encoding="utf-8")

            known = {s["name"] for s in stations}
            missing_from_csv = [o for o in portal_options if o not in known]
            missing_from_portal = [s["name"] for s in todo if s["name"] not in portal_options]

            print(f"\nPortal lists {len(portal_options)} stations for {city}.")
            if missing_from_csv:
                print(f"  {len(missing_from_csv)} portal stations not in stations.csv (extra stations, or the same site under a different agency label):")
                for o in missing_from_csv:
                    print("     ", o)
            if missing_from_portal:
                print(f"  {len(missing_from_portal)} stations.csv entries NOT offered by the portal under that exact name (will be tried anyway):")
                for o in missing_from_portal:
                    print("     ", o)

        except Exception as e:
            print(f"WARNING: couldn't list portal stations for {city} ({e}).")
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass

        # ---- download ----
        for i, station in enumerate(todo, 1):

            print(f"\n[{city} {i}/{len(todo)}]")

            try:
                count = download_station(page, state, portal_city, station["name"], station["folder"])
                results[(city, station["name"])] = count

            except Exception as e:
                print(f"âœ— Station '{station['name']}' failed: {e}")
                results[(city, station["name"])] = 0
                try:
                    page.keyboard.press("Escape")
                except Exception:
                    pass

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("                    COMPLETE")
    print("=" * 65)

    for (city, name), count in results.items():
        flag = "" if count == len(MONTHS) else "   <-- incomplete, rerun to retry"
        print(f"[{city}] {name}: {count}/{len(MONTHS)} months{flag}")

    print("=" * 65)

    if not AUTO:
        input("\nPress ENTER to close browser...")

    browser.close()

