import os
import time
import requests
import json
from urllib.parse import urlparse

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyic", "state")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}

TIMEOUT = 30

# Publicly available Indian government policy PDFs organised by state.
# Each entry: (filename, direct-download URL)
# Sources include official state government portals, NITI Aayog, and ministry sites.
POLICY_URLS = {
    "Andhra_Pradesh": [
        ("AP_IT_Policy_2021-24.pdf",
         "https://it.ap.gov.in/wp-content/uploads/2021/12/AP-IT-Policy-2021-24.pdf"),
        ("AP_Industrial_Policy_2020-23.pdf",
         "https://apindustries.gov.in/APIndus/Data/AP%20Industrial%20Development%20Policy%202020-23.pdf"),
        ("AP_Electronics_Policy_2021-24.pdf",
         "https://it.ap.gov.in/wp-content/uploads/2021/12/AP-Electronics-Policy-2021-24.pdf"),
    ],
    "Arunachal_Pradesh": [
        ("Arunachal_IT_Policy.pdf",
         "https://arunachalpradesh.gov.in/pdf/IT_Policy.pdf"),
        ("Arunachal_Industrial_Policy_2020.pdf",
         "https://arunachalpradesh.gov.in/pdf/Industrial_Policy_2020.pdf"),
    ],
    "Assam": [
        ("Assam_IT_Policy_2017.pdf",
         "https://it.assam.gov.in/sites/default/files/swf_utility_folder/departments/it_com_oid_2/menu/document/Assam_IT_Policy_2017.pdf"),
        ("Assam_Industrial_Policy_2019.pdf",
         "https://industries.assam.gov.in/sites/default/files/swf_utility_folder/departments/industries_com_oid_6/portlet/level_1/Assam_Industrial_Policy_2019.pdf"),
    ],
    "Bihar": [
        ("Bihar_IT_Policy_2017.pdf",
         "https://state.bihar.gov.in/biharit/uploads/IT_Policy_2017.pdf"),
        ("Bihar_Industrial_Policy_2016.pdf",
         "https://state.bihar.gov.in/industries/uploads/Bihar_Industrial_Investment_Promotion_Policy_2016.pdf"),
        ("Bihar_Start_Up_Policy_2017.pdf",
         "https://startup.bihar.gov.in/resources/Bihar_Start_Up_Policy_2017.pdf"),
    ],
    "Chhattisgarh": [
        ("CG_IT_Policy_2019.pdf",
         "https://chips.gov.in/sites/default/files/IT_Policy_2019.pdf"),
        ("CG_Industrial_Policy_2019-24.pdf",
         "https://industries.cg.gov.in/pdf/Industrial_Policy_2019-24.pdf"),
    ],
    "Goa": [
        ("Goa_IT_Policy_2018.pdf",
         "https://www.goa.gov.in/wp-content/uploads/2018/06/Goa-IT-Policy-2018.pdf"),
        ("Goa_Startup_Policy_2017.pdf",
         "https://www.goa.gov.in/wp-content/uploads/2017/09/Goa-Startup-Policy-2017.pdf"),
    ],
    "Gujarat": [
        ("Gujarat_IT_Policy_2022-27.pdf",
         "https://dst.gujarat.gov.in/pdf/Gujarat-IT-ITeS-Policy-2022-27.pdf"),
        ("Gujarat_Industrial_Policy_2020.pdf",
         "https://ic.gujarat.gov.in/uploads/policy/Gujarat_Industrial_Policy_2020.pdf"),
        ("Gujarat_Startup_Policy.pdf",
         "https://startup.gujarat.gov.in/assets/pdf/Startup_Policy.pdf"),
    ],
    "Haryana": [
        ("Haryana_IT_Policy_2017.pdf",
         "https://haryanait.gov.in/uploads/documents/Haryana_IT_Policy_2017.pdf"),
        ("Haryana_Enterprise_Policy_2020.pdf",
         "https://haryanaindustries.gov.in/uploads/Enterprise_Promotion_Policy_2020.pdf"),
    ],
    "Himachal_Pradesh": [
        ("HP_IT_Policy_2019.pdf",
         "https://himachalit.gov.in/uploads/policies/IT_Policy_2019.pdf"),
        ("HP_Industrial_Policy_2019.pdf",
         "https://emerginghimachal.hp.gov.in/themes/developer/document/Industrial_Policy_2019.pdf"),
    ],
    "Jharkhand": [
        ("Jharkhand_IT_Policy_2021.pdf",
         "https://jharkhandit.gov.in/sites/default/files/IT_Policy_2021.pdf"),
        ("Jharkhand_Industrial_Policy_2021.pdf",
         "https://jharkhandindustry.gov.in/sites/default/files/Industrial_Policy_2021.pdf"),
    ],
    "Karnataka": [
        ("Karnataka_IT_Policy_2020-25.pdf",
         "https://itbt.karnataka.gov.in/uploads/media-to-upload/Karnataka_IT_Policy_2020-25.pdf"),
        ("Karnataka_Industrial_Policy_2020-25.pdf",
         "https://kum.karnataka.gov.in/KUM/document/Karnataka_Industrial_Policy_2020-25.pdf"),
        ("Karnataka_Startup_Policy_2022-27.pdf",
         "https://startup.karnataka.gov.in/assets/docs/Karnataka_Startup_Policy_2022-27.pdf"),
    ],
    "Kerala": [
        ("Kerala_IT_Policy_2017.pdf",
         "https://icetkp.org/wp-content/uploads/2017/02/Kerala-IT-Policy-2017.pdf"),
        ("Kerala_Industrial_Policy_2018.pdf",
         "https://keralaindustry.org/images/policy/Kerala_Industrial_Policy_2018.pdf"),
        ("Kerala_Startup_Policy.pdf",
         "https://startupmission.kerala.gov.in/sites/default/files/Kerala_Startup_Policy.pdf"),
    ],
    "Madhya_Pradesh": [
        ("MP_IT_Policy_2016.pdf",
         "https://mpsedc.mp.gov.in/uploads/IT_Policy_2016.pdf"),
        ("MP_Industrial_Policy_2014.pdf",
         "https://mpindustry.gov.in/uploads/Industrial_Policy_2014.pdf"),
    ],
    "Maharashtra": [
        ("Maharashtra_IT_Policy_2023.pdf",
         "https://dit.maharashtra.gov.in/uploads/Maharashtra_IT_Policy_2023.pdf"),
        ("Maharashtra_Industrial_Policy_2019.pdf",
         "https://maitri.mahaonline.gov.in/PDF/Industrial_Policy_2019.pdf"),
        ("Maharashtra_Startup_Policy.pdf",
         "https://startup.maharashtra.gov.in/assets/pdf/Maharashtra_Startup_Policy.pdf"),
    ],
    "Manipur": [
        ("Manipur_IT_Policy.pdf",
         "https://manipurit.gov.in/uploads/IT_Policy.pdf"),
        ("Manipur_Industrial_Policy.pdf",
         "https://manipurindustries.gov.in/uploads/Industrial_Policy.pdf"),
    ],
    "Meghalaya": [
        ("Meghalaya_IT_Policy_2018.pdf",
         "https://meghalayait.gov.in/assets/pdf/IT_Policy_2018.pdf"),
        ("Meghalaya_Industrial_Policy_2016.pdf",
         "https://meghalayaindustries.gov.in/uploads/Industrial_Policy_2016.pdf"),
    ],
    "Mizoram": [
        ("Mizoram_IT_Policy.pdf",
         "https://dit.mizoram.gov.in/uploads/IT_Policy.pdf"),
        ("Mizoram_Industrial_Policy_2019.pdf",
         "https://industries.mizoram.gov.in/uploads/Industrial_Policy_2019.pdf"),
    ],
    "Nagaland": [
        ("Nagaland_IT_Policy_2018.pdf",
         "https://ditc.nagaland.gov.in/uploads/IT_Policy_2018.pdf"),
        ("Nagaland_Industrial_Policy.pdf",
         "https://industries.nagaland.gov.in/uploads/Industrial_Policy.pdf"),
    ],
    "Odisha": [
        ("Odisha_IT_Policy_2022.pdf",
         "https://ict.odisha.gov.in/sites/default/files/2022/IT_Policy_2022.pdf"),
        ("Odisha_Industrial_Policy_2022.pdf",
         "https://investodisha.gov.in/download/Industrial_Policy_Resolution_2022.pdf"),
        ("Odisha_Startup_Policy_2016.pdf",
         "https://startup.odisha.gov.in/assets/pdf/Startup_Policy_2016.pdf"),
    ],
    "Punjab": [
        ("Punjab_IT_Policy_2017.pdf",
         "https://pbdit.gov.in/uploads/policies/IT_Policy_2017.pdf"),
        ("Punjab_Industrial_Policy_2017.pdf",
         "https://pbindustries.gov.in/uploads/Industrial_Policy_2017.pdf"),
    ],
    "Rajasthan": [
        ("Rajasthan_IT_Policy_2015.pdf",
         "https://doitc.rajasthan.gov.in/WriteReadData/Portal/Images/IT_Communication_Policy_2015.pdf"),
        ("Rajasthan_Industrial_Policy_2019.pdf",
         "https://industries.rajasthan.gov.in/content/dam/industries/pdf/RIICO/Industrial_Development_Policy_2019.pdf"),
        ("Rajasthan_Startup_Policy_2015.pdf",
         "https://startup.rajasthan.gov.in/Content/documents/Startup_Policy_2015.pdf"),
    ],
    "Sikkim": [
        ("Sikkim_IT_Policy_2016.pdf",
         "https://sikkim.gov.in/media/pdf/IT_Policy_2016.pdf"),
        ("Sikkim_Industrial_Policy_2017.pdf",
         "https://sikkim.gov.in/media/pdf/Industrial_Policy_2017.pdf"),
    ],
    "Tamil_Nadu": [
        ("TN_IT_Policy_2018.pdf",
         "https://it.tn.gov.in/sites/default/files/TN_IT_Policy_2018.pdf"),
        ("TN_Industrial_Policy_2021.pdf",
         "https://tnindustry.gov.in/pdf/Industrial_Policy_2021.pdf"),
        ("TN_Startup_Policy_2018.pdf",
         "https://startuptn.in/assets/pdf/TN_Startup_Policy_2018.pdf"),
    ],
    "Telangana": [
        ("Telangana_IT_Policy_2021.pdf",
         "https://it.telangana.gov.in/wp-content/uploads/2021/05/Telangana-IT-Policy-2021-Framework.pdf"),
        ("Telangana_Industrial_Policy_2015.pdf",
         "https://tsiic.telangana.gov.in/pdf/TS_I_Policy_2015_21.pdf"),
        ("Telangana_Innovation_Policy_2016.pdf",
         "https://it.telangana.gov.in/wp-content/uploads/2016/11/Telangana-Innovation-Policy-2016.pdf"),
    ],
    "Tripura": [
        ("Tripura_IT_Policy_2017.pdf",
         "https://tripura.gov.in/sites/default/files/IT_Policy_2017.pdf"),
        ("Tripura_Industrial_Policy_2017.pdf",
         "https://industries.tripura.gov.in/sites/default/files/Industrial_Policy_2017.pdf"),
    ],
    "Uttar_Pradesh": [
        ("UP_IT_Policy_2022.pdf",
         "https://uplc.up.nic.in/pdf/UP_IT_Policy_2022.pdf"),
        ("UP_Industrial_Policy_2017.pdf",
         "https://invest.up.gov.in/wp-content/themes/flavor/pdf/UP_Industrial_Investment_Employment_Promotion_Policy_2017.pdf"),
    ],
    "Uttarakhand": [
        ("Uttarakhand_IT_Policy_2018.pdf",
         "https://itda.uk.gov.in/uploads/IT_Policy_2018.pdf"),
        ("Uttarakhand_Industrial_Policy_2015.pdf",
         "https://doiuk.org/uploads/Industrial_Policy_2015.pdf"),
    ],
    "West_Bengal": [
        ("WB_IT_Policy_2018.pdf",
         "https://wbifms.gov.in/pdf/IT_Policy_2018.pdf"),
        ("WB_Industrial_Policy_2013.pdf",
         "https://wbindustry.gov.in/uploaded_files/Industrial_Policy_2013.pdf"),
    ],
    "Andaman_and_Nicobar_Islands": [
        ("ANI_Industrial_Policy.pdf",
         "https://andaman.gov.in/downloads/Industrial_Policy.pdf"),
    ],
    "Chandigarh": [
        ("Chandigarh_IT_Policy_2015.pdf",
         "https://chandigarh.gov.in/sites/default/files/IT_Policy_2015.pdf"),
        ("Chandigarh_Startup_Policy_2018.pdf",
         "https://chandigarh.gov.in/sites/default/files/Startup_Policy_2018.pdf"),
    ],
    "Dadra_and_Nagar_Haveli_and_Daman_and_Diu": [
        ("DNHDD_Industrial_Policy.pdf",
         "https://ddd.gov.in/uploads/Industrial_Policy.pdf"),
    ],
    "Delhi": [
        ("Delhi_Startup_Policy_2016.pdf",
         "https://startup.delhi.gov.in/content/dam/doitassets/pdf/Delhi_Startup_Policy_2016.pdf"),
        ("Delhi_Industrial_Policy_2010.pdf",
         "https://delhigovt.nic.in/wps/wcm/connect/delhigovtresources/Industrial_Policy_2010.pdf"),
    ],
    "Jammu_and_Kashmir": [
        ("JK_IT_Policy_2020.pdf",
         "https://jkit.nic.in/uploads/IT_Policy_2020.pdf"),
        ("JK_Industrial_Policy_2021.pdf",
         "https://jkindustrialpolicy.nic.in/pdfs/JK_New_Industrial_Policy_2021.pdf"),
    ],
    "Ladakh": [
        ("Ladakh_UT_Policy_Framework.pdf",
         "https://ladakh.gov.in/uploads/Policy_Framework.pdf"),
    ],
    "Lakshadweep": [
        ("Lakshadweep_Admin_Policy.pdf",
         "https://lakshadweep.gov.in/uploads/Admin_Policy.pdf"),
    ],
    "Puducherry": [
        ("Puducherry_IT_Policy_2018.pdf",
         "https://dit.py.gov.in/sites/default/files/IT_Policy_2018.pdf"),
        ("Puducherry_Industrial_Policy_2016.pdf",
         "https://industries.py.gov.in/sites/default/files/Industrial_Policy_2016.pdf"),
    ],
}


def download_pdf(url: str, save_path: str) -> bool:
    """Download a single PDF and return True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=True)
        content_type = resp.headers.get("Content-Type", "")

        if resp.status_code != 200:
            print(f"  [SKIP] HTTP {resp.status_code}: {url}")
            return False

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(save_path)
        if file_size < 1024:
            os.remove(save_path)
            print(f"  [SKIP] File too small ({file_size}B), likely not a PDF: {url}")
            return False

        print(f"  [OK] Saved: {save_path} ({file_size / 1024:.1f} KB)")
        return True

    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] {url}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  [CONN_ERROR] {url}")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}: {url}")
        return False


def main():
    total = 0
    success = 0
    failed = 0
    results = {}

    for state, policies in sorted(POLICY_URLS.items()):
        print(f"\n{'='*60}")
        print(f"State: {state.replace('_', ' ')}")
        print(f"{'='*60}")
        state_success = 0

        for filename, url in policies:
            total += 1
            save_path = os.path.join(BASE_DIR, state, filename)

            if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                print(f"  [EXISTS] {filename}")
                success += 1
                state_success += 1
                continue

            print(f"  Downloading: {filename}")
            if download_pdf(url, save_path):
                success += 1
                state_success += 1
            else:
                failed += 1

            time.sleep(0.5)

        results[state] = state_success

    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total attempts : {total}")
    print(f"Successful     : {success}")
    print(f"Failed/Skipped : {failed}")
    print(f"{'='*60}")

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyic", "download_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "total_attempted": total,
            "successful": success,
            "failed": failed,
            "per_state": results,
        }, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
