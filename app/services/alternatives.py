"""
app/services/alternatives.py
The Eatlytic "Ingredient-Pivot" Alternative Engine. 
Converts product categories, names, and risk flags into behavioral healthier swaps.
"""
import logging

logger = logging.getLogger(__name__)

def get_healthy_alternative(product_category: str, persona: str = "general", product_name: str = "", ingredients: str = "") -> str:
    """
    Takes the category and returns an ingredient-pivot suggestion.
    Tailored for the Indian market and specific personas (Diabetic, Athlete, Seniors, Pregnant, Kids, Keto, Skin).
    Optimized to dynamically customize the recommendation based on the product name and ingredients.
    """
    
    # Standardize inputs
    p_cat = str(product_category or "").lower().strip()
    p_pers = str(persona or "general").lower().strip()
    p_name = str(product_name or "").lower().strip()
    p_ings = str(ingredients or "").lower().strip()

    # 1. Custom Dynamic overrides based on product name and ingredients
    if "butter" in p_name:
        if "keto" in p_pers:
            return "🥗 Keto Option: Butter is excellent for Keto! Use in moderation, preferably choosing grass-fed organic butter or home-made white butter (Makhan)."
        if "senior" in p_pers or "heart" in p_pers:
            return "🥗 Heart Health Swap: Saturated fats should be limited. Swap butter for extra virgin olive oil or high-oleic cold-pressed oils for cardiovascular support."
        return "🥗 Healthy Swap: Choose grass-fed pasteurized butter or A2 cow ghee. Keep daily intake under 1-2 teaspoons."

    if "salt" in p_name:
        if "senior" in p_pers or "heart" in p_pers or "diabetic" in p_pers:
            return "🥗 Salt Swap: High sodium spikes blood pressure. Swap regular refined salt with low-sodium potassium salt or natural rock salt (Saindhav salt) in extremely limited quantities."
        return "🥗 Healthy Swap: Switch from refined iodized salt to unrefined pink Himalayan salt or rock salt, which contains natural trace minerals."

    if "maggi" in p_name or "noodle" in p_name:
        if "diabetic" in p_pers:
            return "🥗 Diabetic Noodle Swap: Strictly avoid instant flour noodles. Switch to shirataki (konjac) noodles, or prepare a high-protein paneer bhurji with fresh vegetables."
        if "keto" in p_pers:
            return "🥗 Keto Noodle Swap: Avoid wheat/rice noodles. Use shirataki noodles or zucchini spirals (zoodles) tossed in olive oil or butter."
        if "kid" in p_pers or "child" in p_pers:
            return "🥗 Kid-Friendly Noodle Swap: Avoid instant noodles containing palm oil and INS 635 flavor enhancers. Serve home-made suji vermicelli or whole wheat millet noodles with fresh vegetables."
        return "🥗 Healthy Noodle Swap: Instead of deep-fried palm-oil instant noodles, try whole wheat vermicelli (Sevai), red rice poha, or baked millet-based noodles cooked with fresh spices and vegetables."

    if "jim jam" in p_name or "biscuit" in p_name or "cookie" in p_name or "oreo" in p_name:
        if "diabetic" in p_pers:
            return "🥗 Diabetic Biscuit Swap: Highly refined maida and sugar biscuits will cause glucose spikes. Avoid them and snack on a handful of roasted almonds, walnuts, or chia seeds."
        if "keto" in p_pers:
            return "🥗 Keto Cookie Swap: Avoid conventional wheat/sugar biscuits. Choose almond flour or coconut flour cookies sweetened with erythritol or stevia."
        if "kid" in p_pers or "child" in p_pers:
            return "🥗 Kid Cookie Swap: Avoid refined flour biscuits containing trans fats, palm oil, or synthetic colors (like INS 122). Swap with home-made ragi-jaggery cookies or high-fiber whole grain crackers."
        return "🥗 Healthy Biscuit Swap: Swap refined flour (Maida) biscuits with whole wheat (Atta) or oat-based biscuits baked with minimal sugar and cold-pressed oils, or fresh fruit."

    if "lays" in p_name or "chip" in p_name or "kurkure" in p_name or "namkeen" in p_name:
        if "diabetic" in p_pers:
            return "🥗 Diabetic Snack Swap: Avoid potato/corn chips that spike blood sugar. Snack on dry-roasted peanuts, baked chickpeas, or pumpkin seeds."
        if "keto" in p_pers:
            return "🥗 Keto Snack Swap: Potato/corn chips are high-carb. Swap with baked cheese crisps, roasted pumpkin seeds, or home-made salted kale chips."
        if "senior" in p_pers or "heart" in p_pers:
            return "🥗 Senior Snack Swap: Avoid high-sodium deep-fried chips. Snack on unsalted roasted makhana (fox nuts), raw almonds, or fresh cucumber slices."
        return "🥗 Healthy Snack Swap: Swap deep-fried palmolein oil chips for dry-roasted makhana, baked beetroot/sweet potato chips, or home-made air-fried vegetable chips."

    if "palm oil" in p_ings or "palmolein" in p_ings:
        alt_suffix = " Swap this product for alternatives made with cold-pressed oils, butter, A2 ghee, or coconut oil to avoid inflammatory fats."
        if "diabetic" in p_pers:
            return "🥗 Anti-Inflammatory Swap: Refined palm oil increases diabetic cardiovascular risk." + alt_suffix
        if "senior" in p_pers or "heart" in p_pers:
            return "🥗 Heart-Healthy Swap: Palm oil is high in palmitic acid which raises LDL cholesterol." + alt_suffix
        return "🥗 Healthy Oil Swap: This product contains refined palm oil." + alt_suffix

    # The Indian Alternative Matrix
    alternatives_map = {
        "biscuit": {
            "default_alt": "Look for biscuits made with Whole Wheat (Atta) or Oats, baked (not fried), with less than 5g added sugar per 100g.",
            "diabetic_alt": "Avoid biscuits entirely. Snack on a handful of roasted almonds or walnuts instead.",
            "keto_alt": "Avoid wheat/sugar biscuits. Choose almond flour or coconut flour keto cookies sweetened with erythritol or stevia.",
            "pregnant_alt": "Avoid biscuits with trans fat or palm oil. Opt for homemade ragi cookies or organic multi-grain snacks.",
            "senior_alt": "Choose low-sodium ragi/digestive biscuits with <3g saturated fat. Check for high palm oil.",
            "kid_alt": "Choose whole wheat or ragi jaggery cookies. Avoid conventional store-bought biscuits with synthetic preservatives.",
            "skin_alt": "Avoid refined flour/sugar biscuits. Snack on seeds (sunflower, pumpkin) or fresh berries to avoid acne-triggering glycation."
        },
        "noodle": {
            "default_alt": "Switch to plain Vermicelli (Sevai) or Oats Poha. Add your own vegetables and salt to control sodium.",
            "diabetic_alt": "Strictly avoid instant noodles. They spike blood sugar faster than table sugar.",
            "keto_alt": "Avoid flour noodles. Use shirataki (konjac) noodles or zucchini spirals (zoodles) with olive oil.",
            "pregnant_alt": "Avoid instant noodles (high sodium/preservatives). Choose whole wheat sewai or red rice poha with boiled peas.",
            "senior_alt": "Avoid high-sodium instant noodles. Prepare freshly made home vermicelli with high veggies and minimal salt.",
            "kid_alt": "Avoid instant noodles with INS 635 / MSG. Serve homemade suji vermicelli or multi-grain noodles with fresh ingredients."
        },
        "chip": {
            "default_alt": "Look for baked snacks, roasted Makhana (fox nuts), or roasted chana. Check for 'Vacuum Fried' on the label.",
            "diabetic_alt": "Choose roasted peanuts or baked chickpeas instead of high-carb potato chips.",
            "keto_alt": "Avoid potato/corn chips. Choose baked cheese crisps, salted pumpkin seeds, or home-made kale chips.",
            "pregnant_alt": "Avoid fried chips high in saturated fats. Snack on lightly salted roasted makhana or baked beetroot chips.",
            "senior_alt": "Avoid salted potato chips (sodium bomb). Choose zero-salt roasted makhana, roasted walnuts, or fresh cucumber slices."
        },
        "beverage": {
            "default_alt": "Switch to plain water, lemon water, or green tea. If you need energy, eat a banana—it's healthier than any energy drink.",
            "athlete_alt": "Look for drinks with Electrolytes and <10g natural sugar, no Maltodextrin.",
            "keto_alt": "Avoid sweetened drinks. Sip on unsweetened green tea, black coffee, or plain sparkling water.",
            "pregnant_alt": "Avoid caffeinated/sugary energy drinks. Choose fresh coconut water, buttermilk (chaas), or home-squeezed lime juice.",
            "senior_alt": "Avoid high-sugar energy drinks. Drink tender coconut water (excellent potassium) or unsweetened green tea."
        },
        "juice": {
            "default_alt": "Eat the whole fruit instead. If you must buy juice, look for '100% Cold Pressed' with ZERO added sugar or preservatives.",
            "diabetic_alt": "Avoid fruit juices entirely. Eat a whole apple or orange to keep the fiber intact and slow sugar absorption.",
            "keto_alt": "Avoid fruit juices (extremely high in fructose). Enjoy a few whole strawberries or raspberries.",
            "pregnant_alt": "Choose fresh home-squeezed juice with the pulp. Avoid commercial juices pasteurized at high heat with preservatives."
        },
        "dairy": {
            "default_alt": "Switch to plain, unsweetened dahi (curd) or milk. Add fresh fruits at home for sweetness.",
            "diabetic_alt": "Choose plain Greek yogurt or unsweetened buttermilk. Avoid sweetened low-fat dairy which is packed with sugar.",
            "keto_alt": "Choose high-fat plain Greek yogurt, heavy cream, or paneer (cottage cheese). Avoid skimmed milk."
        },
        "chocolate": {
            "default_alt": "Switch to Dark Chocolate (>70% Cocoa) with less than 15g sugar per 100g. Avoid 'Milk Chocolate' with hydrogenated fats.",
            "diabetic_alt": "Choose dark chocolate (>85% cocoa) or stevia-sweetened chocolate. Keep portions extremely small (1-2 squares).",
            "keto_alt": "Choose dark chocolate (>90% cocoa) or sugar-free stevia chocolate. High fat, very low carb.",
            "kid_alt": "Avoid conventional milk chocolate high in emulsifiers and sugar. Opt for 70% dark chocolate in small quantities."
        },
        "protein_supplement": {
            "default_alt": "Look for 'Whey Protein Isolate' with <3g carbs/sugar per scoop, and no Maltodextrin in the ingredients.",
            "athlete_alt": "Ensure it has a complete Amino Acid profile. Avoid 'Protein Blends' that hide cheap fillers."
        },
        "ready_to_eat": {
            "default_alt": "Look for 'Retort Processed' meals without preservatives. Switch to fresh home-cooked Dal-Chawal for better fiber.",
            "diabetic_alt": "Avoid commercial ready-to-eat meals due to refined carbs and thickeners. Opt for quick home-made paneer bhurji.",
            "senior_alt": "Avoid ready-to-eat foods due to high sodium/preservatives. Eat fresh, warm, home-cooked dal and soft khichdi."
        },
        "sweet": {
            "default_alt": "Enjoy traditional sweets made with Jaggery or Dates in moderation. Avoid those with 'Liquid Glucose' or Artificial Colors.",
            "diabetic_alt": "Strictly avoid conventional sweets. Opt for diabetic-friendly desserts sweetened with stevia or erythritol in moderation."
        }
    }

    # 2. Check if we have a rule for this category
    category_data = alternatives_map.get(p_cat)
    
    if not category_data:
        return "🥗 Healthier Swap: Choose whole foods with <5 ingredients, no added sugar, and <400mg sodium per 100g."

    # 3. Check for Persona-Specific Overrides
    if "diabetic" in p_pers and "diabetic_alt" in category_data:
        return f"🥗 Alternative (For Diabetics): {category_data['diabetic_alt']}"

    if ("athlete" in p_pers or "gym" in p_pers) and "athlete_alt" in category_data:
        return f"🥗 Alternative (For Athletes): {category_data['athlete_alt']}"

    if "keto" in p_pers and "keto_alt" in category_data:
        return f"🥗 Alternative (For Keto): {category_data['keto_alt']}"

    if ("preg" in p_pers or "matern" in p_pers) and "pregnant_alt" in category_data:
        return f"🥗 Alternative (For Pregnancy): {category_data['pregnant_alt']}"

    if ("senior" in p_pers or "60+" in p_pers or "heart" in p_pers) and "senior_alt" in category_data:
        return f"🥗 Alternative (For Heart & Seniors): {category_data['senior_alt']}"

    if ("child" in p_pers or "kid" in p_pers or "parent" in p_pers) and "kid_alt" in category_data:
        return f"🥗 Alternative (For Kids): {category_data['kid_alt']}"

    if "skin" in p_pers and "skin_alt" in category_data:
        return f"🥗 Alternative (For Skin Health): {category_data['skin_alt']}"

    # 4. Default category alternative
    return f"🥗 Healthier Swap: {category_data['default_alt']}"
