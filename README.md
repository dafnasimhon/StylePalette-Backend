# StylePalette-Backend
Python Backend API for StylePalette

🎨 Color Analysis Engine: The Seasonal Method
To provide personalized outfit recommendations, StyleMate implements a modular analysis engine based on the professional Seasonal Color Theory. This engine processes the extracted features (Skin, Hair, and Eye color) through three distinct logical tests to classify the user into one of the four primary "Color Seasons."

1. The Temperature Test (Undertone)
The primary objective of this test is to determine the user's Undertone. In color science, skin, hair, and eyes are categorized as having either a Warm (yellow/golden base) or Cool (blue/pink base) bias.

Warm: Often associated with ginger or golden-blonde hair and tan skin. These users are best complemented by earth tones and golden hues.

Cool: Characterized by ash-toned or black hair and fair or olive skin. These users shine in jewel tones and silver-based colors.

Implementation: The engine uses a weighted scoring system (Heuristic) to calculate the cumulative temperature of all user features.

2. The Contrast Test
This test measures the Visual Intensity of the user's appearance. Contrast is determined by the "value gap" between the lightness of the skin and the darkness of the hair and eyes.

High Contrast: A significant difference (e.g., Very Fair skin with Black hair). These users can carry bold, saturated, and high-intensity outfits.

Low Contrast: Features that are close in depth (e.g., Fair skin with Blonde hair). These users are better suited for softer, monochromatic, or tonal color palettes.

3. The Chroma & Value Test
This final stage refines the classification by analyzing the Purity (Chroma) and Depth (Value) of the colors.

Value (Depth): Determines if the user is "Light" or "Deep." A "Deep" user (Dark hair/eyes) looks best in rich, heavy colors (e.g., Burgundy, Navy), while a "Light" user looks best in airy, bright colors.

Chroma (Saturation): Distinguishes between Clear (bright and striking) and Muted (soft and grayish) features. This is the deciding factor between "Winter" (Clear/Cool) and "Summer" (Muted/Cool).