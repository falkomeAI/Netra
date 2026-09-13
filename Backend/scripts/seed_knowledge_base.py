#!/usr/bin/env python3
"""
Seed the NETRA knowledge base with Indian government scheme data.
Cleans existing garbage entries and loads curated scheme information.

Usage:
    python3 scripts/seed_knowledge_base.py          # Clean + seed
    python3 scripts/seed_knowledge_base.py --clean   # Only clean garbage
    python3 scripts/seed_knowledge_base.py --append   # Seed without cleaning
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "knowledge_base" / "vector_store"
META_PATH = KB_DIR / "metadata.json"
INDEX_PATH = KB_DIR / "index.faiss"

GOVERNMENT_SCHEMES = [
    {
        "title": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
        "content": """PM-KISAN Yojana - Pradhan Mantri Kisan Samman Nidhi:
PM-KISAN is a central government scheme that provides income support of Rs 6,000 per year to all landholding farmer families across India.
The amount is paid in three equal installments of Rs 2,000 each, directly into the farmer's bank account.
Eligibility: All landholding farmer families with cultivable land. Small and marginal farmers are the primary beneficiaries.
Who cannot apply: Institutional land holders, farmer families holding constitutional posts, serving or retired government employees drawing pension above Rs 10,000/month, income tax payers, professionals like doctors, engineers, lawyers, chartered accountants.
How to apply: Visit the nearest Common Service Center (CSC) or apply online at pmkisan.gov.in. Required documents: Aadhaar card, bank account details, land ownership documents.
Helpline: 155261 or 011-24300606.
The money comes automatically to your bank account. You don't need to do anything after registration — it will come in 3 installments every 4 months."""
    },
    {
        "title": "MGNREGA (Mahatma Gandhi National Rural Employment Guarantee Act)",
        "content": """MGNREGA - Mahatma Gandhi National Rural Employment Guarantee Act:
MGNREGA guarantees 100 days of wage employment per year to every rural household whose adult members volunteer to do unskilled manual work.
Daily wage: Rs 267-333 per day (varies by state). In FY 2025-26, the average wage is approximately Rs 289 per day.
Eligibility: Any adult member of a rural household willing to do unskilled manual work. No land ownership required.
How to apply: Go to your Gram Panchayat office with a passport-size photo. They will issue a Job Card within 15 days. The Job Card is free of cost.
Key rights: If work is not provided within 15 days of applying, you are entitled to unemployment allowance. Work must be provided within 5 km of your village. At least one-third of workers must be women. Wages must be paid within 15 days through bank/post office account.
Required documents: Aadhaar card, photo, bank account. Job Card number for tracking.
Helpline: 1800-345-22-44 (toll free). You can also check your payment status at nrega.nic.in."""
    },
    {
        "title": "Ayushman Bharat - PM Jan Arogya Yojana (PMJAY)",
        "content": """Ayushman Bharat - Pradhan Mantri Jan Arogya Yojana (PMJAY):
Ayushman Bharat provides health insurance cover of Rs 5 lakh per family per year for secondary and tertiary care hospitalization.
It is the world's largest government-funded health insurance scheme covering over 50 crore beneficiaries.
Eligibility: Based on SECC (Socio Economic Caste Census) 2011 data. Deprived rural families and identified occupational categories in urban areas. No restriction on family size, age, or gender.
Benefits: Cashless and paperless treatment at any empanelled hospital (public or private) across India. Covers 1,929 medical procedures including surgery, medical treatment, and day care.
Pre-existing conditions are covered from day one. No cap on family size.
How to check eligibility: Visit mera.pmjay.gov.in or call 14555. Enter your Aadhaar number or ration card number.
How to get the card: Visit your nearest Ayushman Bharat Arogya Mitra at any empanelled hospital or Common Service Center with Aadhaar card and ration card.
Helpline: 14555 (toll free) or 1800-111-565."""
    },
    {
        "title": "Ration Card / National Food Security Act (NFSA)",
        "content": """Ration Card - National Food Security Act (NFSA):
Under the National Food Security Act, eligible households get subsidized food grains through the Public Distribution System (PDS).
Benefits: AAY (Antyodaya) card holders get 35 kg of food grains per family per month at Rs 2/kg wheat, Rs 3/kg rice, Rs 1/kg coarse grains.
Priority Household (PHH) card holders get 5 kg per person per month at the same subsidized rates.
PM Garib Kalyan Anna Yojana (PMGKAY): Additional 5 kg free food grains per person per month.
Eligibility: Below Poverty Line (BPL) families, Antyodaya (poorest of poor) families. Based on state-specific criteria.
How to apply: Apply at your district Food & Civil Supplies office or through state food portal. Required: Aadhaar card, address proof, income certificate, family photograph.
To check ration card status: Visit nfsa.gov.in or your state's food department website.
Helpline: 1967 or 1800-345-7777 (varies by state)."""
    },
    {
        "title": "PM Fasal Bima Yojana (Crop Insurance)",
        "content": """Pradhan Mantri Fasal Bima Yojana (PMFBY) - Crop Insurance:
PMFBY provides insurance coverage and financial support to farmers in case of crop failure due to natural calamities, pests, and diseases.
Premium rates: Kharif crops: 2% of sum insured. Rabi crops: 1.5% of sum insured. Horticulture/commercial crops: 5% of sum insured.
The remaining premium is paid by the government (center and state share equally).
Coverage: All food crops, oilseeds, horticulture, and commercial crops. Covers prevented sowing, mid-season adversity, post-harvest losses (up to 14 days), and localized calamities.
Eligibility: All farmers including sharecroppers and tenant farmers growing notified crops. Both loanee and non-loanee farmers can enroll.
How to apply: Through your bank branch (if you have a crop loan), Common Service Center, or the PMFBY portal at pmfby.gov.in. Apply before the cutoff date for each crop season.
Required documents: Aadhaar card, bank account, land records (khasra/khatauni), sowing certificate from village officer.
Helpline: 1800-200-7710 (toll free). Crop loss must be reported within 72 hours."""
    },
    {
        "title": "Soil Health Card Scheme",
        "content": """Soil Health Card Scheme:
The Soil Health Card scheme provides information to farmers about the nutrient status of their soil along with recommendations on appropriate dosage of nutrients for improving soil health and fertility.
Benefits: Free soil testing and nutrient recommendations. Helps reduce unnecessary fertilizer costs and improve crop yields by 10-15%.
The card includes: soil type, macronutrients (Nitrogen, Phosphorus, Potassium), micronutrients (Iron, Zinc, Copper, Manganese, Boron), pH level, Electrical Conductivity, Organic Carbon.
Based on the test results, specific crop-wise fertilizer recommendations are provided.
How to get: Contact your local Krishi Vigyan Kendra (KVK), agricultural department office, or Common Service Center. Soil samples are collected by trained personnel.
Soil Health Card is issued once every 2 years for each farm holding.
Check your card online at soilhealth.dac.gov.in.
Helpline: Kisan Call Center 1800-180-1551 (toll free)."""
    },
    {
        "title": "Kisan Credit Card (KCC)",
        "content": """Kisan Credit Card (KCC):
KCC provides farmers with affordable credit for agricultural needs including crop cultivation, post-harvest expenses, farm maintenance, and consumption needs.
Credit limit: Based on land holding, cropping pattern, and scale of finance. Can go up to Rs 3 lakh at subsidized interest rate.
Interest rate: 7% per annum with 3% subvention for timely repayment, making effective rate just 4% per annum for loans up to Rs 3 lakh.
Eligibility: All farmers — individual/joint borrowers, owner cultivators, tenant farmers, sharecroppers, and self-help groups.
Also covers animal husbandry and fisheries activities.
How to apply: Visit your nearest bank (any commercial bank, cooperative bank, or regional rural bank) with: Aadhaar card, PAN card, land records, passport-size photos, and application form.
The bank must process the application within 14 days.
Crop insurance under PMFBY is also available through KCC.
Helpline: Contact your bank or Kisan Call Center 1800-180-1551."""
    },
    {
        "title": "PM Awas Yojana - Gramin (PMAY-G) Housing Scheme",
        "content": """PM Awas Yojana - Gramin (PMAY-G):
PMAY-G provides financial assistance to rural families for construction of pucca houses with basic amenities.
Assistance amount: Rs 1,20,000 in plain areas and Rs 1,30,000 in hilly/difficult areas. Additional Rs 12,000 for toilet construction under Swachh Bharat Mission.
Beneficiaries also get 90-95 person-days of unskilled labor under MGNREGA.
Eligibility: Houseless families, families living in kutcha or dilapidated houses. Selection based on SECC 2011 data and verified by Gram Sabha. Priority: SC/ST, freed bonded laborers, minorities, differently abled, families with no literate adult member, single women-headed households.
How to apply: Selected beneficiaries are identified from SECC data through Gram Sabha verification. Check your name at pmayg.nic.in using your Aadhaar number.
The amount is transferred directly to the beneficiary's bank account in installments linked to construction stages, verified through geo-tagged photos.
Helpline: 1800-11-6446 (toll free)."""
    },
    {
        "title": "Sukanya Samriddhi Yojana (Girl Child Savings)",
        "content": """Sukanya Samriddhi Yojana (SSY):
SSY is a government savings scheme for the girl child under the Beti Bachao Beti Padhao campaign.
Interest rate: 8.2% per annum (one of the highest among government small savings schemes).
Minimum deposit: Rs 250 per year. Maximum: Rs 1,50,000 per year.
Account can be opened for a girl child below 10 years of age. Only 2 accounts allowed per family (one per daughter).
Maturity: 21 years from the date of opening. Partial withdrawal of 50% allowed after the girl turns 18 for education.
Tax benefits: Deposits eligible for deduction under Section 80C. Interest and maturity amount are tax-free (EEE status).
How to open: Visit any post office or authorized bank (SBI, PNB, BOB, etc.) with: birth certificate of girl child, Aadhaar card of parent/guardian, address proof, photographs.
Helpline: Contact your nearest post office or bank."""
    },
    {
        "title": "PM Ujjwala Yojana (Free LPG Connection)",
        "content": """Pradhan Mantri Ujjwala Yojana (PMUY):
PMUY provides free LPG connections to women from Below Poverty Line (BPL) families to reduce health hazards from cooking with firewood and cow dung.
Benefits: Free LPG connection, free first refill cylinder, and free stove. Deposit-free LPG connection.
Under Ujjwala 2.0: No address proof required — a self-declaration is sufficient.
Eligibility: Women belonging to BPL households, SC/ST households, PMAY beneficiaries, most backward classes, tea/forest dwellers, people living in islands and river islands, and SECC-listed households.
How to apply: Visit your nearest LPG distributor (HP, Bharat, Indane) with: Aadhaar card, BPL ration card or SECC list confirmation, bank account details, passport-size photograph.
First refill: Free. Subsequent refills at subsidized rates with subsidy credited to bank account.
Helpline: 1800-266-6696 (toll free)."""
    },
    {
        "title": "PM Vishwakarma Yojana (for Artisans and Craftspeople)",
        "content": """PM Vishwakarma Yojana:
This scheme supports traditional artisans and craftspeople working with their hands and tools across 18 trades.
Covered trades: Carpenter, boat maker, blacksmith, hammer and toolkit maker, locksmith, goldsmith, potter, sculptor, cobbler, mason, basket/mat maker, doll/toy maker, barber, garland maker, washerman, tailor, fishing net maker, and armourer.
Benefits: Recognition through PM Vishwakarma certificate and ID card. Skill training: basic (5-7 days) and advanced (15 days) with Rs 500/day stipend. Toolkit incentive of Rs 15,000. Collateral-free credit: Rs 1 lakh (first tranche) and Rs 2 lakh (second tranche) at 5% interest. Marketing support through quality certification, branding, and e-commerce linkage.
Eligibility: Artisans/craftspeople aged 18+ working in one of the 18 trades. Must register through Common Service Center or pm-vishwakarma.gov.in.
Required documents: Aadhaar card, bank account, mobile number, trade certificate (self-declaration accepted).
Helpline: 1800-599-3957 (toll free)."""
    },
    {
        "title": "Jan Dhan Yojana (Bank Account for All)",
        "content": """Pradhan Mantri Jan Dhan Yojana (PMJDY):
PMJDY ensures every Indian household has access to a basic bank account with a zero-balance requirement.
Benefits: Zero balance savings account. Free RuPay debit card with Rs 2 lakh accident insurance cover. Overdraft facility of Rs 10,000 (for eligible accounts). Life insurance cover of Rs 30,000 (for accounts opened before January 2015). Free mobile banking facility.
Direct Benefit Transfer (DBT): All government scheme payments (PM-KISAN, MGNREGA wages, LPG subsidy, scholarships) are transferred directly to Jan Dhan accounts.
Eligibility: Any Indian citizen who doesn't have a bank account. Aadhaar-based simplified KYC.
How to open: Visit any bank branch or Banking Correspondent with Aadhaar card. No minimum balance required.
Helpline: 1800-11-0001 or 1800-180-1111."""
    },
    {
        "title": "Pradhan Mantri Gram Sadak Yojana (PMGSY) - Village Road Development",
        "content": """Pradhan Mantri Gram Sadak Yojana (PMGSY) - Village Road Development:
PMGSY provides all-weather road connectivity to unconnected habitations in rural areas.
Phase I: Connects habitations with 500+ population (250+ in hilly, tribal, desert areas) with all-weather roads.
Phase II: Upgrades existing rural roads that connect markets, schools, and hospitals.
Phase III (launched 2019): Consolidates 1,25,000 km of routes connecting habitations to Gramin Agricultural Markets (GrAMs), higher secondary schools, and hospitals.
Road specifications: Two-lane roads with proper drainage. Built to withstand local traffic and weather conditions.
Village impact: Better access to markets for selling crops, access to healthcare and education, reduced travel time, emergency vehicle access.
How villages benefit: Gram Panchayat can request road under PMGSY through Block Development Officer (BDO). Road proposals are prioritized based on habitation population and connectivity needs.
Citizens can track road construction progress at omms.nic.in (Online Management, Monitoring & Accounting System).
Helpline: Contact your District Rural Development Agency (DRDA) or Block Development Office."""
    },
    {
        "title": "Swachh Bharat Mission - Gramin (Village Sanitation)",
        "content": """Swachh Bharat Mission - Gramin (SBM-G):
SBM-G aims to make villages Open Defecation Free (ODF) by constructing household and community toilets.
Phase I: Rs 12,000 per household for Individual Household Latrine (IHHL) construction.
Phase II (SBM-G Phase II): Focus on ODF sustainability and Solid/Liquid Waste Management (SLWM) in villages.
Benefits: Free toilet construction with government subsidy. Community sanitary complexes in public places. Solid waste management (composting, segregation). Liquid waste management (soak pits, waste stabilization ponds, constructed wetlands).
Village cleanliness components: Door-to-door waste collection, plastic waste management, grey water management, fecal sludge management, and biodegradable waste composting.
Eligibility: All BPL households, SC/ST households, small and marginal farmers, landless laborers, physically handicapped, and women-headed households.
How to apply: Apply through your Gram Panchayat or Block Development Office. Register at sbm.gov.in.
GOBARDHAN scheme: Converts cattle dung and biodegradable waste into biogas and bio-CNG, providing clean cooking fuel and extra income.
Helpline: 1969 or visit sbm.gov.in."""
    },
    {
        "title": "PM Kusum Yojana (Solar Energy for Farmers)",
        "content": """Pradhan Mantri Kisan Urja Suraksha evam Utthan Mahabhiyan (PM-KUSUM):
PM-KUSUM promotes solar energy in agriculture to reduce diesel/electricity costs for farmers.
Component A: 10,000 MW of decentralized solar power plants on barren/fallow land. Farmers can earn Rs 60,000-1,00,000 per acre per year by setting up solar plants.
Component B: Installation of 20 lakh standalone solar-powered agriculture pumps to replace diesel pumps. Subsidy: 30% from Central Government + 30% from State Government. Farmer pays only 40%.
Component C: Solarization of 15 lakh existing grid-connected agriculture pumps. Farmers can sell surplus solar power to DISCOMs and earn extra income.
Eligibility: Individual farmers, groups of farmers, cooperatives, Panchayats, Farmer Producer Organizations.
Benefits: Reduced electricity bills, guaranteed income from solar power sale, reliable irrigation water supply, environment-friendly farming.
How to apply: Apply through your state's renewable energy department or mnre.gov.in. Contact the local agriculture office or DISCOM.
Helpline: 1800-180-3333 (MNRE toll-free)."""
    },
    {
        "title": "National Rural Livelihood Mission (DAY-NRLM) - Self Help Groups",
        "content": """Deendayal Antyodaya Yojana - National Rural Livelihood Mission (DAY-NRLM):
DAY-NRLM organizes rural poor women into Self Help Groups (SHGs) for livelihood improvement.
Self Help Groups (SHGs): Groups of 10-20 women from similar economic backgrounds who save regularly and give loans to each other.
Benefits: Revolving fund of Rs 10,000-15,000 per SHG. Community Investment Fund (CIF) up to Rs 2.5 lakh. Bank linkage: SHGs can get loans from banks at low interest rates.
Interest subvention: Women SHGs get loans at 7% interest, with additional 3% subvention for timely repayment (effective 4%).
Skill development and livelihood support. Marketing support through SHG products.
Village Organization (VO): Federation of 10-15 SHGs at village level. Cluster Level Federation (CLF): Federation of VOs.
How to join: Contact your village's ASHA worker, Anganwadi worker, or Block Development Office. Or visit nrlm.gov.in.
SHG Didi: Women SHG members also serve as Banking Correspondents, providing banking services in villages without bank branches.
Helpline: 011-23461708 or contact your State Rural Livelihood Mission."""
    },
    {
        "title": "Jal Jeevan Mission (Har Ghar Jal - Piped Water)",
        "content": """Jal Jeevan Mission (JJM) - Har Ghar Jal:
JJM aims to provide piped drinking water (Functional Household Tap Connection - FHTC) to every rural household by 2024.
Benefits: Clean tap water at 55 liters per person per day. Regular water quality testing. Water quality labs at district and block level.
Village Water and Sanitation Committee (VWSC) or Pani Samiti: Manages village water supply system, collects user charges, ensures maintenance.
5% of the project cost is contributed by the village community. 10% in hilly and forested areas.
In-village infrastructure: Piped water supply scheme with source, treatment plant, overhead tank, and household tap connections.
Grey water management is mandatory under JJM.
Water quality: Bureau of Indian Standards (BIS) 10500 compliant. Regular testing through Field Test Kits (FTK) by village community.
How to get connection: Contact your Gram Panchayat or VWSC. Free for all households (connection charge borne by government).
Track progress: ejalshakti.gov.in/jjmreport
Helpline: Contact your District Jal Jeevan Mission office or Public Health Engineering Department (PHED)."""
    },
    {
        "title": "PM Matsya Sampada Yojana (Fisheries Development)",
        "content": """Pradhan Mantri Matsya Sampada Yojana (PMMSY):
PMMSY promotes fish farming and aquaculture with financial assistance for construction of ponds, cages, and hatcheries.
Benefits for village fish farmers:
- Construction of new ponds: 40-60% subsidy (60% for SC/ST/Women). Rs 7 lakh per hectare for freshwater ponds.
- Biofloc/RAS fish farming: Up to 40% subsidy on unit cost.
- Ornamental fish culture: 40% subsidy for backyard ornamental fish units.
- Fish feed mills: 40% subsidy for setting up fish feed production.
- Cold chain and market infrastructure: Insulated vehicles, ice plants, fish kiosks.
- Fishing boats and nets: Subsidy for motorization and deep-sea fishing vessels.
Eligibility: Fishers, fish farmers, fish workers, fish vendors, SHGs, cooperatives, entrepreneurs in fisheries sector.
Inland fisheries: Focus on reservoirs, ponds, tanks, wetlands, rivers. Cage culture in reservoirs. Paddy-cum-fish culture.
How to apply: Apply through your District Fisheries Officer or at pmmsy.dof.gov.in. Applications usually through state fisheries department.
Helpline: 1800-425-1660 or contact your District Fisheries Officer."""
    },
    {
        "title": "PM Kisan Maandhan Yojana (Farmer Pension)",
        "content": """Pradhan Mantri Kisan Maandhan Yojana (PM-KMY) - Farmer Pension:
PM-KMY is a voluntary pension scheme for small and marginal farmers.
Pension: Rs 3,000 per month after age 60. If the farmer dies, spouse gets 50% (Rs 1,500/month) as family pension.
Monthly contribution: Rs 55-200 per month (based on entry age). Entry age 18 years: Rs 55/month. Entry age 30 years: Rs 100/month. Entry age 40 years: Rs 200/month.
The government contributes an equal matching amount.
Eligibility: Small and marginal farmers aged 18-40 with cultivable land up to 2 hectares. Should not be covered under NPS, ESIC, or EPFO. Should not be an income taxpayer.
How to enroll: Visit your nearest Common Service Center (CSC) with Aadhaar card, bank passbook, and land records. Or register at maandhan.in.
Auto-debit: Monthly contribution is deducted automatically from the farmer's bank account.
Helpline: 1800-267-6888 (toll free) or 14434."""
    },
    {
        "title": "Samagra Shiksha Abhiyan (Village Education)",
        "content": """Samagra Shiksha Abhiyan - Integrated Education Scheme for Villages:
Samagra Shiksha covers school education from pre-primary to Class 12, with focus on improving quality and access in rural areas.
Key benefits for village students:
- Free textbooks to all students in government schools.
- Free uniforms: Rs 600 per child per year.
- Mid-Day Meal (PM POSHAN): Free cooked nutritious meal for all students in Classes 1-8.
- Transport/Escort allowance for students in areas without nearby schools.
- Residential hostel facilities for remote areas.
- Kasturba Gandhi Balika Vidyalaya (KGBV): Residential schools for girls from SC/ST/OBC/minority and BPL families in educationally backward blocks.
- Smart classrooms with ICT infrastructure.
- Bridge courses for out-of-school children to re-enter mainstream education.
- Inclusive education: Special support for children with disabilities — assistive devices, home-based education, special educators.
Village-level School Management Committee (SMC) ensures community participation.
How to enroll: Contact your nearest government school or Cluster Resource Centre (CRC). Education is free and compulsory under Right to Education Act for ages 6-14.
Helpline: Contact your Block Education Officer or District Education Officer."""
    },
    {
        "title": "Pradhan Mantri Mudra Yojana (PMMY) - Small Business Loans",
        "content": """Pradhan Mantri Mudra Yojana (PMMY) - Loans for Small Business:
PMMY provides collateral-free loans up to Rs 10 lakh for small businesses and enterprises.
Three categories:
- Shishu: Loans up to Rs 50,000 (for starting businesses). Interest: ~10-12% per annum.
- Kishore: Loans from Rs 50,001 to Rs 5 lakh (for growing businesses).
- Tarun: Loans from Rs 5,00,001 to Rs 10 lakh (for established businesses seeking expansion).
No collateral or guarantor required. No processing fee.
Eligible activities: Manufacturing, trading, service sector, agriculture-allied activities (dairy, poultry, beekeeping, fishery, food processing, etc.).
Village entrepreneurs: Street vendors, artisans, tailors, shopkeepers, fruit/vegetable sellers, mechanics, salon owners, papad/pickle makers, rickshaw owners.
How to apply: Visit any commercial bank, RRB (Regional Rural Bank), cooperative bank, MFI (Micro Finance Institution), or NBFC. Apply with: Aadhaar card, PAN card, business plan, address proof, photographs.
Mudra Card: A debit card facility for working capital requirements.
Helpline: 1800-180-1111 or visit mudra.org.in."""
    },
]


def clean_knowledge_base():
    """Remove garbage entries from the existing KB."""
    if not META_PATH.exists():
        print("No existing metadata — nothing to clean.")
        return

    metadata = json.loads(META_PATH.read_text())
    original_count = len(metadata)

    garbage_indicators = ["You", "mystery", "oyster"]

    cleaned = []
    for entry in metadata:
        text = entry.get("text", "").strip()
        if len(text) < 5:
            continue
        if text in garbage_indicators:
            continue
        ascii_junk = sum(1 for c in text if ord(c) > 0x10000 or c in "\ufffd")
        if ascii_junk > len(text) * 0.3:
            continue
        words = text.split()
        if len(words) < 3:
            continue
        cleaned.append(entry)

    removed = original_count - len(cleaned)
    print(f"Cleaned KB: removed {removed} garbage entries, kept {len(cleaned)} valid entries.")

    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
        print("Deleted old FAISS index (will be rebuilt on next embedding run).")

    META_PATH.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2))
    return cleaned


def seed_schemes():
    """Add government scheme data to KB via the embedding pipeline."""
    print(f"\nSeeding {len(GOVERNMENT_SCHEMES)} government schemes into knowledge base...")

    from netra.config.settings import Settings
    from netra.core.pipeline import Pipeline
    from netra.utils.log import setup_logging

    config_dir = PROJECT_ROOT / "config"
    settings = Settings(config_dir)
    setup_logging(level="WARNING")

    import netra.stages  # noqa: F401
    pipeline = Pipeline(settings)

    for i, scheme in enumerate(GOVERNMENT_SCHEMES, 1):
        text = scheme["content"]
        inputs = {
            "text": text,
            "ocr_output": {"text": text, "lines": text.split("\n"), "num_lines": len(text.split("\n"))},
            "target_language": "hi",
            "source_language": "en",
        }

        try:
            results = pipeline.run(inputs, stages=["embedding", "knowledge_graph"])
            emb_result = results.get("embedding")
            if emb_result and emb_result.success:
                chunks = emb_result.data.get("num_chunks", 1)
                print(f"  [{i}/{len(GOVERNMENT_SCHEMES)}] {scheme['title']} — {chunks} chunk(s) indexed")
            else:
                err = emb_result.error if emb_result else "unknown"
                print(f"  [{i}/{len(GOVERNMENT_SCHEMES)}] {scheme['title']} — FAILED: {err}")
        except Exception as e:
            print(f"  [{i}/{len(GOVERNMENT_SCHEMES)}] {scheme['title']} — ERROR: {e}")

    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else []
    print(f"\nDone! Knowledge base now has {len(meta)} chunks total.")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--clean" in args:
        clean_knowledge_base()
    elif "--append" in args:
        seed_schemes()
    else:
        clean_knowledge_base()
        seed_schemes()
