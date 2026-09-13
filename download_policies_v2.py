import os
import time
import requests
import json

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyic", "state")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}

TIMEOUT = 60

# Verified working URLs for Indian state government policy PDFs (as of Aug 2026)
POLICY_URLS = {
    "Andhra_Pradesh": [
        ("AP_IT_GCC_Policy_4.0_2024-2029.pdf",
         "https://apiic.in/wp-content/themes/custom-theme/assets/Pdfs/AP%20IT%20&%20GCC%20Policy%20(4%200)%202024-2029_%20G%20O%20MS%20No.9,%20dt%2012%2012%202024.pdf"),
        ("AP_Industrial_Development_Policy_4.0_2024-29.pdf",
         "https://apiic.in/wp-content/uploads/2024/12/GOMS-NO-68.pdf"),
    ],
    "Arunachal_Pradesh": [
        ("Arunachal_Pradesh_Industrial_Development_Investment_Policy_2025.pdf",
         "https://indarun.gov.in/pdf/Arunachal%20Pradesh%20Industrial%20Development%20and%20Investment%20Policy,%202025.pdf"),
    ],
    "Assam": [
        ("Assam_Industrial_Investment_Policy_Amendment_2023.pdf",
         "https://industries.assam.gov.in/sites/default/files/swf_utility_folder/departments/industries_com_oid_6/portlet/level_1/industrial_and_investment_policy_of_assam_amendment_2023.pdf"),
    ],
    "Bihar": [
        ("Bihar_Industrial_Investment_Promotion_Policy_2016.pdf",
         "https://static.investindia.gov.in/s3fs-public/2020-02/Bihar_Industrial_Investment_Promotion_Policy_2016.pdf"),
    ],
    "Chhattisgarh": [
        ("CG_Industrial_Development_Policy_2024-30.pdf",
         "https://oneclick.cgstate.gov.in/cmsadmin/storage/app/uploads/policy/Industrial%20Policy%20Notification/Industrial%20Policy%202024-30/Industrial%20Development%20Policy%202024-30%20%28As%20amended%20on%2027.05.2025%29.pdf"),
        ("CG_Industrial_Development_Policy_2024-30_Brief.pdf",
         "https://oneclick.cgstate.gov.in/cmsadmin/storage/app/uploads/policy/Policy%20&%20Act/industrial%20policy%202024-30/IDP_2024-30(Brief).pdf"),
        ("CG_Innovation_Startup_Promotion_Policy_2025-2030.pdf",
         "https://invest.cg.gov.in/storage/pdfs/Chhattisgarh%20Innovation%20and%20Startup%20Promotion%20Policy%202025-2030.pdf"),
    ],
    "Goa": [
        ("Goa_Industrial_Growth_Investment_Promotion_Policy_2022.pdf",
         "https://www.goaipb.goa.gov.in/wp-content/uploads/2024/01/Goa-Industrial-Growth-And-Investment-Promotion-Policy-2022.pdf"),
    ],
    "Gujarat": [
        ("Gujarat_Industrial_Policy_2020.pdf",
         "https://ic.gujarat.gov.in/uploads/policy/Gujarat_Industrial_Policy_2020.pdf"),
    ],
    "Haryana": [
        ("Haryana_Make_In_Haryana_Industrial_Policy_2025.pdf",
         "https://investharyana.in/content/pdfs/Make%20in%20Haryana%20Industrial%20Policy%202025%20-%2015.09.2025.pdf"),
    ],
    "Himachal_Pradesh": [
        ("HP_Industrial_Policy_2019.pdf",
         "https://static.investindia.gov.in/s3fs-public/2020-02/Himachal_Pradesh_Industrial_Policy_2019.pdf"),
    ],
    "Jharkhand": [
        ("Jharkhand_Industrial_Investment_Policy_2021.pdf",
         "https://static.investindia.gov.in/s3fs-public/2021-06/Jharkhand_Industrial_and_Investment_Promotion_Policy_2021.pdf"),
    ],
    "Karnataka": [
        ("Karnataka_Industrial_Policy_2025-30.pdf",
         "https://investkarnataka.co.in/wp-content/uploads/2025/02/IndustrialPolicy2025_PrintPagesSingle_.pdf"),
        ("Karnataka_Industrial_Policy_Operative_Guidelines_2025-30.pdf",
         "https://investkarnataka.co.in/wp-content/uploads/2026/04/Karnataka.pdf"),
    ],
    "Kerala": [
        ("Kerala_IT_Policy_2026.pdf",
         "https://ibyinfopark.in/public/pdf/kerala-it-policy-2026.pdf"),
    ],
    "Madhya_Pradesh": [
        ("MP_Industrial_Promotion_Policy_2025.pdf",
         "https://static.investindia.gov.in/s3fs-public/2025-05/ipp.pdf"),
    ],
    "Maharashtra": [
        ("Maharashtra_Startup_Entrepreneurship_Innovation_Policy_2025.pdf",
         "https://msins.in/assets/Maharashtra-Startup-Entrepreneurship-_-Innovation-Policy-2025-C5OBUYrd.pdf"),
    ],
    "Manipur": [
        ("Manipur_Industrial_Investment_Promotion_Policy_2022.pdf",
         "https://static.investindia.gov.in/s3fs-public/2022-06/Manipur_Industrial_Investment_Promotion_Policy_2022.pdf"),
    ],
    "Meghalaya": [
        ("Meghalaya_Industrial_Investment_Promotion_Policy_2024.pdf",
         "https://static.investindia.gov.in/s3fs-public/2024-05/meghalaya_industrial_and_investment_promotion_policy_2024.pdf"),
    ],
    "Mizoram": [
        ("Mizoram_Industrial_Investment_Policy_2021.pdf",
         "https://static.investindia.gov.in/s3fs-public/2021-06/Mizoram_Industrial_and_Investment_Policy_2021.pdf"),
    ],
    "Nagaland": [
        ("Nagaland_Industrial_Policy.pdf",
         "https://static.investindia.gov.in/s3fs-public/2020-02/Nagaland_Industrial_Policy.pdf"),
    ],
    "Odisha": [
        ("Odisha_Industrial_Policy_Resolution_2022.pdf",
         "https://investodisha.gov.in/download/Industrial_Policy_Resolution_2022.pdf"),
    ],
    "Punjab": [
        ("Punjab_Industrial_Business_Development_Policy_2022.pdf",
         "https://static.investindia.gov.in/s3fs-public/2023-01/Punjab_Industrial_and_Business_Development_Policy_2022.pdf"),
    ],
    "Rajasthan": [
        ("Rajasthan_Investment_Promotion_Scheme_RIPS_2024.pdf",
         "https://finance.rajasthan.gov.in/PDFDOCS/TAX/CCT/14514.pdf"),
        ("Rajasthan_RIPS_2024_Notification.pdf",
         "https://epch.in/sites/default/files/policies/RIPS-2024-Notified-on-8-10-2024.pdf"),
    ],
    "Sikkim": [
        ("Sikkim_Industrial_Investment_Policy_2024.pdf",
         "https://static.investindia.gov.in/s3fs-public/2024-03/sikkim_industrial_and_investment_policy_2024.pdf"),
    ],
    "Tamil_Nadu": [
        ("TN_Industrial_Policy_2021.pdf",
         "https://static.investindia.gov.in/s3fs-public/2021-02/Tamil%20Nadu%20Industrial%20Policy%202021%20%281%29.pdf"),
        ("TN_Industries_Policy_Note_2024-25.pdf",
         "https://cms.tn.gov.in/cms_migrated/document/docfiles/ind_e_pn_2024_25.pdf"),
        ("TN_Industries_Policy_Note_2025-26.pdf",
         "https://cms.tn.gov.in/cms_migrated/document/docfiles/ind_e_pn_2025_26.pdf"),
    ],
    "Telangana": [
        ("Telangana_ICT_Policy_Framework_2016.pdf",
         "https://it.telangana.gov.in/wp-content/uploads/2016/04/Telangana-ICT-Policy-Framework-2016.pdf"),
        ("Telangana_AI_Framework_2020.pdf",
         "https://it.telangana.gov.in/wp-content/uploads/2020/07/Govt-of-Telangana-Artificial-Intelligence-Framework-2020.pdf"),
        ("Telangana_AI_Powered_Strategy_2024.pdf",
         "https://it.telangana.gov.in/wp-content/uploads/2024/09/AI-Powered-Telangana-Strategy-Document-and-Implementation-Roadmap.pdf"),
        ("Telangana_IoT_Policy_2017.pdf",
         "https://it.telangana.gov.in/wp-content/uploads/2017/10/Telangana-IoT-Policy-2017.pdf"),
    ],
    "Tripura": [
        ("Tripura_Industrial_Investment_Policy_2024.pdf",
         "https://static.investindia.gov.in/s3fs-public/2024-01/tripura_industrial_investment_promotion_incentive_scheme_2022.pdf"),
    ],
    "Uttar_Pradesh": [
        ("UP_Industrial_Investment_Employment_Promotion_Policy_2022.pdf",
         "https://static.investindia.gov.in/s3fs-public/2024-12/uttar_pradesh_industrial_investment_employment_promotion_policy_2022-en.pdf"),
    ],
    "Uttarakhand": [
        ("Uttarakhand_Industrial_Policy_2015.pdf",
         "https://static.investindia.gov.in/s3fs-public/2020-02/Uttarakhand_Industrial_Investment_and_Employment_Promotion_Policy_2015.pdf"),
    ],
    "West_Bengal": [
        ("WB_Industrial_Policy_2013.pdf",
         "https://static.investindia.gov.in/s3fs-public/2020-02/West_Bengal_Incentive_Scheme_2013.pdf"),
    ],
    "Delhi": [
        ("Delhi_Startup_Policy_2016.pdf",
         "https://static.investindia.gov.in/s3fs-public/2020-02/Delhi_Startup_Policy.pdf"),
    ],
    "Jammu_and_Kashmir": [
        ("JK_New_Industrial_Policy_2021-30.pdf",
         "https://static.investindia.gov.in/s3fs-public/2021-04/JK_New_Industrial_Policy_2021-30.pdf"),
    ],
    "Chandigarh": [
        ("Chandigarh_Industrial_Policy.pdf",
         "https://static.investindia.gov.in/s3fs-public/2020-02/Chandigarh_Industrial_Policy.pdf"),
    ],
    "Puducherry": [
        ("Puducherry_Industrial_Policy.pdf",
         "https://static.investindia.gov.in/s3fs-public/2020-02/Puducherry_Industrial_Promotion_Policy.pdf"),
    ],
    # Central / NITI Aayog
    "Andaman_and_Nicobar_Islands": [
        ("NITI_Aayog_Annual_Report_2024-25.pdf",
         "https://niti.gov.in/sites/default/files/2025-02/Annual%20Report%202024-25%20English_FINAL_LOW%20RES_0.pdf"),
    ],
}


def download_pdf(url: str, save_path: str) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=True)

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
            print(f"  [SKIP] File too small ({file_size}B), likely error page: {url}")
            return False

        print(f"  [OK] {save_path} ({file_size / 1024:.1f} KB)")
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

            time.sleep(0.3)

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
