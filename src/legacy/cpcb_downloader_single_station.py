from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

STATE = "Delhi"
CITY = "Delhi"
STATION = "Alipur, Delhi - DPCC"
YEAR = "2023"

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

DOWNLOAD_FOLDER = os.path.abspath("CPCB_Data")


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

    input("Press ENTER after solving the CAPTCHA... ")

    # Give Angular some time to remove the overlay
    page.wait_for_timeout(2000)

    # Wait until CAPTCHA modal disappears
    try:
        page.locator("#captchaModal").wait_for(
            state="hidden",
            timeout=10000
        )
        print("✓ CAPTCHA modal closed.")
    except PlaywrightTimeoutError:
        print("WARNING: CAPTCHA modal may still be present.")
        print("Please make sure it is closed before continuing.")
        input("Press ENTER to continue anyway... ")

    # Wait for loading overlay to disappear
    try:
        page.locator("#loader-overlay").wait_for(
            state="hidden",
            timeout=15000
        )
        print("✓ CPCB loading overlay removed.")
    except PlaywrightTimeoutError:
        print("WARNING: Loader overlay is still present.")


# ============================================================
# WAIT UNTIL PAGE IS READY
# ============================================================

def wait_until_ready(page):

    print("Waiting for CPCB page to become ready...")

    # CAPTCHA
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

    print("✓ Page ready.")


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

        for i in range(min(options.count(), 30)):
            try:
                print(
                    "   ",
                    options.nth(i).inner_text()
                )
            except:
                pass

        raise Exception(
            f"Could not find option '{value}' "
            f"for {name}"
        )

    # Exact match if possible
    exact_option = page.locator(
        ".ng-option"
    ).filter(
        has_text=value
    )

    exact_option.first.click()

    page.wait_for_timeout(1000)

    print(f"✓ {name} selected.")


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

    print("✓ Submit clicked.")

    # Wait for results
    page.wait_for_timeout(3000)

    wait_until_ready(page)


# ============================================================
# DOWNLOAD MONTH
# ============================================================

def download_month(page, month):

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

        print(f"✗ Could not find row for {month}")

        return False

    # Find button in row
    buttons = matching_row.locator("button")

    if buttons.count() == 0:

        print("No button found, checking links...")

        buttons = matching_row.locator("a")

    if buttons.count() == 0:

        print(f"✗ No download button for {month}")

        return False

    button = buttons.last

    try:

        with page.expect_download(
            timeout=30000
        ) as download_info:

            button.click()

        download = download_info.value

        filename = download.suggested_filename

        # Add month if CPCB gives a generic filename
        save_path = os.path.join(
            DOWNLOAD_FOLDER,
            filename
        )

        download.save_as(save_path)

        print(f"✓ Downloaded: {filename}")

        return True

    except Exception as e:

        print(f"✗ Download failed:")
        print(e)

        return False


# ============================================================
# MAIN
# ============================================================

with sync_playwright() as p:

    # --------------------------------------------------------
    # CREATE DOWNLOAD DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        DOWNLOAD_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # START BROWSER
    # --------------------------------------------------------

    browser = p.chromium.launch(
        headless=False
    )

    context = browser.new_context(
        accept_downloads=True
    )

    page = context.new_page()

    page.set_default_timeout(15000)

    # --------------------------------------------------------
    # OPEN CPCB
    # --------------------------------------------------------

    print()
    print("Opening CPCB AQI Repository...")

    page.goto(
        "https://airquality.cpcb.gov.in/ccr/#/repository/aqi",
        wait_until="domcontentloaded"
    )

    print("Page loaded.")

    # --------------------------------------------------------
    # CAPTCHA
    # --------------------------------------------------------

    wait_for_captcha(page)

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    select_dropdown(
        page,
        1,
        STATE,
        "State"
    )

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    select_dropdown(
        page,
        2,
        CITY,
        "City"
    )

    # --------------------------------------------------------
    # STATION
    # --------------------------------------------------------

    select_dropdown(
        page,
        3,
        STATION,
        "Station"
    )

    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    select_dropdown(
        page,
        4,
        YEAR,
        "Year"
    )

    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    click_submit(page)

    # --------------------------------------------------------
    # DOWNLOAD MONTHS
    # --------------------------------------------------------

    successful = 0

    for month in MONTHS:

        result = download_month(
            page,
            month
        )

        if result:
            successful += 1

        time.sleep(2)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("                    COMPLETE")
    print("=" * 65)

    print(f"State    : {STATE}")
    print(f"City     : {CITY}")
    print(f"Station  : {STATION}")
    print(f"Year     : {YEAR}")

    print(
        f"Downloaded: "
        f"{successful}/{len(MONTHS)} months"
    )

    print()
    print(
        f"Files saved in:\n"
        f"{DOWNLOAD_FOLDER}"
    )

    print("=" * 65)

    input("\nPress ENTER to close browser...")

    browser.close()