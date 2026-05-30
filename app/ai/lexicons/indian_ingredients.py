"""
app/ai/lexicons/indian_ingredients.py
─────────────────────────────────────────────────────────────────────────────
Eatlytic Indian Ingredient & Additive Lexicon

Maps ingredient names / INS codes to:
  - Glycemic Index (GI) level and typical GI value
  - Safety tier for diabetic persons
  - Clinical reasoning and curiosity facts

This is PURE DATA — no imports, no side effects, no business logic.
It is consumed by the EatlyticBrain service to perform deterministic,
offline nutritional audits without any LLM calls.

Coverage:
  - Starches / Flours / High Glycemic Carbs
  - Fats / Oils (Refined, Hydrogenated)
  - Sweeteners (Natural, Artificial, Sugar Alcohols)
  - Flavor Enhancers (MSG, Ribonucleotides)
  - Synthetic Food Dyes (Azo dyes, Coal-tar derivatives)
"""

INDIAN_INGREDIENT_LEXICON: dict = {
    # ── Starches / Flours / High Glycemic Carbs ──────────────────────────
    "maltodextrin": {
        "name": "Maltodextrin",
        "category": "starch",
        "gi_level": "HIGH",
        "typical_gi": 110,
        "diabetic_verdict": "AVOID",
        "reason": "Maltodextrin has a Glycemic Index of 110-185—spiking blood sugar faster than pure table glucose (GI ~100). Highly dangerous for insulin resistance.",
        "curiosity_fact": "It is widely used in packaged foods as a cheap filler, thickener, and preservative.",
        "ins_codes": ["ins 1400", "e1400"]
    },
    "maida": {
        "name": "Maida (Refined Wheat Flour)",
        "category": "flour",
        "gi_level": "HIGH",
        "typical_gi": 75,
        "diabetic_verdict": "AVOID",
        "reason": "Extremely refined flour with fiber and nutrients stripped. Causes rapid insulin surges and contributes to visceral fat accumulation.",
        "curiosity_fact": "Refined flour is chemically bleached with benzoyl peroxide or chlorine gas in industrial milling.",
        "ins_codes": []
    },
    "refined wheat flour": {
        "name": "Refined Wheat Flour (Maida)",
        "category": "flour",
        "gi_level": "HIGH",
        "typical_gi": 75,
        "diabetic_verdict": "AVOID",
        "reason": "Essentially Maida. High starch content with zero fibrous buffer to slow glucose absorption.",
        "curiosity_fact": "The bran and germ are completely removed, stripping >90% of dietary fiber, vitamins, and minerals.",
        "ins_codes": []
    },
    "corn starch": {
        "name": "Corn Starch / Corn Flour",
        "category": "starch",
        "gi_level": "HIGH",
        "typical_gi": 85,
        "diabetic_verdict": "AVOID",
        "reason": "Pure refined carbohydrate that acts as a rapid glycemic spike trigger. Highly concentrated.",
        "curiosity_fact": "Mainly used as a thickener and binder in gravies and instant noodles.",
        "ins_codes": []
    },
    "potato starch": {
        "name": "Potato Starch",
        "category": "starch",
        "gi_level": "HIGH",
        "typical_gi": 80,
        "diabetic_verdict": "AVOID",
        "reason": "Highly refined starch with a rapid rate of digestion, promoting fast postprandial glucose rises.",
        "curiosity_fact": "Has high binding power and yields high viscosity when cooked.",
        "ins_codes": []
    },
    "dextrose": {
        "name": "Dextrose (D-Glucose)",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 100,
        "diabetic_verdict": "AVOID",
        "reason": "Pure medical glucose. Absorbs directly through the stomach lining, causing an immediate, severe glycemic surge.",
        "curiosity_fact": "Structurally identical to the glucose circulating in human blood vessels.",
        "ins_codes": []
    },
    "maltose": {
        "name": "Maltose (Malt Sugar)",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 105,
        "diabetic_verdict": "AVOID",
        "reason": "A disaccharide composed of two glucose units. Higher glycemic impact than standard sucrose.",
        "curiosity_fact": "Produced during the malting of barley grains.",
        "ins_codes": []
    },
    "liquid glucose": {
        "name": "Liquid Glucose / Glucose Syrup",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 90,
        "diabetic_verdict": "AVOID",
        "reason": "Highly concentrated glucose solution that leaves zero digestive resistance, triggering immediate insulin secretion.",
        "curiosity_fact": "Extensively used in Indian confectionery to prevent sugar crystallization.",
        "ins_codes": []
    },
    "invert sugar": {
        "name": "Invert Sugar / Invert Syrup",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 65,
        "diabetic_verdict": "AVOID",
        "reason": "A mixture of glucose and fructose. Very sweet, causes high liver-glycogen stress and glycemic load.",
        "curiosity_fact": "Sweeter than regular sugar and keeps baked goods moist for much longer.",
        "ins_codes": []
    },
    "high fructose corn syrup": {
        "name": "High Fructose Corn Syrup (HFCS)",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 65,
        "diabetic_verdict": "AVOID",
        "reason": "High fructose load drives hepatic lipogenesis (fatty liver), systemic inflammation, and severe leptin resistance.",
        "curiosity_fact": "Fructose is processed exclusively by the liver, unlike glucose which can be used by all cells.",
        "ins_codes": []
    },
    "sugar": {
        "name": "Refined Sugar (Sucrose)",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 65,
        "diabetic_verdict": "AVOID",
        "reason": "Consists of 50% glucose and 50% fructose. High empty calorie profile. Direct cause of pancreatic fatigue and chronic inflammation.",
        "curiosity_fact": "Sugar triggers the release of dopamine in the brain's reward center, highly mimicking addictive substances.",
        "ins_codes": []
    },

    # ── Fats / Oils ──────────────────────────────────────────────────────
    "palm oil": {
        "name": "Refined Palm Oil",
        "category": "oil",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Highly refined, industrially processed fat. High in palmitic acid which is highly pro-inflammatory and increases LDL cardiovascular risk in diabetics.",
        "curiosity_fact": "Palm oil is the most widely consumed vegetable oil in the world due to its low cost and high melting point.",
        "ins_codes": []
    },
    "palmolein": {
        "name": "Refined Palmolein Oil",
        "category": "oil",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Identical concern to palm oil. Liquid fraction of palm oil, loaded with industrial saturated fatty acids that worsen metabolic markers.",
        "curiosity_fact": "Often used for commercial deep frying because it can withstand repeated heating without breaking down.",
        "ins_codes": []
    },
    "hydrogenated vegetable oil": {
        "name": "Hydrogenated Vegetable Oil / Vanaspati",
        "category": "oil",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "AVOID",
        "reason": "Source of industrial trans fats. Wrecks lipid panels, spikes systemic inflammation, damages blood vessel linings, and raises stroke risk.",
        "curiosity_fact": "Vegetable oil is bubbled with hydrogen gas at high temperatures in the presence of a nickel catalyst to solidify it.",
        "ins_codes": []
    },

    # ── Sweeteners (Artificial / Sugar Alcohols) ─────────────────────────
    "aspartame": {
        "name": "Aspartame",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Artificial sweetener. Zero glycemic index but can trigger insulin responses via cephalic phase activation. Classified as possibly carcinogenic (Group 2B) by IARC.",
        "curiosity_fact": "It is about 200 times sweeter than sucrose and breaks down at high baking temperatures.",
        "ins_codes": ["ins 951", "e951", "951"]
    },
    "acesulfame potassium": {
        "name": "Acesulfame Potassium (Acesulfame K)",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Synthetic artificial sweetener. Low-calorie but linked to alterations in gut microbiota. Often blended with aspartame.",
        "curiosity_fact": "Discovered accidentally in 1967 when a chemist licked his finger.",
        "ins_codes": ["ins 950", "e950", "950"]
    },
    "sucralose": {
        "name": "Sucralose",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Chlorinated sugar derivative. Zero glycemic, but evidence suggests chronic use can decrease insulin sensitivity and alter gut biome.",
        "curiosity_fact": "It is made by selectively substituting three chlorine atoms for three hydroxyl groups on sucrose.",
        "ins_codes": ["ins 955", "e955", "955"]
    },
    "steviol glycosides": {
        "name": "Stevia (Steviol Glycosides)",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "SAFE",
        "reason": "Natural, plant-derived non-nutritive sweetener. Zero glycemic index, zero calories, and safe for metabolic health.",
        "curiosity_fact": "Extracted from the leaves of the Stevia rebaudiana plant, native to South America.",
        "ins_codes": ["ins 960", "e960", "960"]
    },
    "stevia": {
        "name": "Stevia Extract",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "SAFE",
        "reason": "Pure plant-based sweetener. Zero glycemic surge, highly recommended for sugar substitution.",
        "curiosity_fact": "Leaves have been used by indigenous South American tribes for over 1,500 years to sweeten teas.",
        "ins_codes": []
    },
    "erythritol": {
        "name": "Erythritol",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "SAFE",
        "reason": "Natural sugar alcohol. Near-zero calories, zero glycemic impact, and highly tolerated without gastrointestinal side effects.",
        "curiosity_fact": "About 90% is absorbed in the small intestine and excreted unchanged in urine.",
        "ins_codes": ["ins 968", "e968", "968"]
    },
    "maltitol": {
        "name": "Maltitol",
        "category": "sweetener",
        "gi_level": "MODERATE",
        "typical_gi": 35,
        "diabetic_verdict": "CAUTION",
        "reason": "Sugar alcohol with a moderate glycemic index (GI ~35). Spikes blood glucose and insulin (about half the impact of sugar).",
        "curiosity_fact": "Frequently used in 'sugar-free' chocolates, but can cause bloating or laxative effects.",
        "ins_codes": ["ins 965", "e965", "965"]
    },
    "sorbitol": {
        "name": "Sorbitol",
        "category": "sweetener",
        "gi_level": "MODERATE",
        "typical_gi": 9,
        "diabetic_verdict": "CAUTION",
        "reason": "Sugar alcohol. Low glycemic index, but can cause severe abdominal cramping and osmotic diarrhea in moderate doses.",
        "curiosity_fact": "Naturally occurs in fruits like apples and pears but is industrially synthesized from corn syrup.",
        "ins_codes": ["ins 420", "e420", "420"]
    },

    # ── Flavor Enhancers ─────────────────────────────────────────────────
    "monosodium glutamate": {
        "name": "Monosodium Glutamate (MSG)",
        "category": "additive",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Umami flavor enhancer. Generally recognized as safe by FSSAI, but can act as an excitotoxin in high amounts, potentially causing headaches.",
        "curiosity_fact": "It is the sodium salt of glutamic acid, an amino acid abundant in nature.",
        "ins_codes": ["ins 621", "e621", "621"]
    },
    "disodium 5'-ribonucleotides": {
        "name": "Disodium 5'-Ribonucleotides",
        "category": "additive",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Synthetic flavor enhancer (INS 635) that synergetically works with MSG to intensify savory taste profiles.",
        "curiosity_fact": "Often found in instant noodle tastemakers, chips, and spice mixes.",
        "ins_codes": ["ins 635", "e635", "635"]
    },

    # ── Synthetic Food Dyes ───────────────────────────────────────────────
    "tartrazine": {
        "name": "Tartrazine (Synthetic Yellow Dye)",
        "category": "dye",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "AVOID",
        "reason": "Coal-tar derived artificial dye. Banned or heavily restricted in several European countries due to links with childhood hyperactivity and hives.",
        "curiosity_fact": "Mandatorily carries a warning label in the EU: 'May have an adverse effect on activity and attention in children.'",
        "ins_codes": ["ins 102", "e102", "102"]
    },
    "sunset yellow": {
        "name": "Sunset Yellow FCF (Orange Dye)",
        "category": "dye",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "AVOID",
        "reason": "Petroleum-derived artificial food coloring. Associated with allergic reactions, worsening asthma, and hyperactivity in kids.",
        "curiosity_fact": "Used in candies, carbonated orange beverages, and packaged snack foods.",
        "ins_codes": ["ins 110", "e110", "110"]
    },
    "carmoisine": {
        "name": "Carmoisine (Red Dye)",
        "category": "dye",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "AVOID",
        "reason": "Synthetic red azo dye. Strongly linked to hives and hyperactivity. Banned in Canada, USA, and Japan.",
        "curiosity_fact": "Often used in jams, gelatin desserts, and packaged pastries.",
        "ins_codes": ["ins 122", "e122", "122"]
    },
}
