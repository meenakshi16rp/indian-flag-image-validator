# Indian Flag Image Validator 

## Description
The **Indian Flag Validator** is a Python-based tool that automatically validates Indian national flags (Tiranga) for color accuracy, stripe proportions, and Ashoka Chakra specifications according to **Bureau of Indian Standards (BIS)**. 

## 🔍 Context
The Indian national flag has strict rules defined by the BIS:

- **Aspect ratio:** 3:2 (width:height)  
- **Horizontal stripes:** Saffron  (#FF9933), White (#FFFFFF) , Green  (#138808) - each 1/3 of height)  
- **Ashoka Chakra:** 24 evenly spaced spokes, centered in the white band, diameter must be 3/4 of white band
- Colors: ±5% RGB tolerance for all colors including Chakra Blue (#000080)

This project validates a given flag image (PNG, JPG, or SVG) and outputs a detailed JSON report showing compliance with BIS standards.

## 🚀  Features
- Validates **aspect ratio**  
- Validates **color accuracy** (Saffron, White, Green, Chakra Blue)  
- Validates **stripe proportions** (each stripe exactly 1/3 of height)
- Detects **Ashoka Chakra**: center position, diameter, and 24 spokes  
- Works for **any resolution** without hardcoded templates  
- Handles **local files or image URLs**  
- Supports **PNG, JPG, and SVG**  
- Processes images ≤5MB within ~3 seconds
- outputs a  JSON report with pass/fail and deviation percentages

## 🧪 Test images
The repository includes 4 test images for validation:

test_1.png: ✅ Perfect BIS-compliant flag --> expected result: all checks pass (PERFECT FLAG)
test_2.png: ✅ Correct but larger resolution --> expected result: all checks pass (DIFFERENT RESOLUTION)
test_3.png: ❌ 4:3 ratio instead of 3:2 --> expected result: Aspect ratio fails (WRONG ASPECT RATIO)
test_4.png: ❌ 20 spokes instead of 24 -->  expected result: Spoke count fails (WRONG SPOKE COUNT)


## Developed for the Indian Independence Day Coding Challenge (MINOR IN AI- IIT ROPAR)
