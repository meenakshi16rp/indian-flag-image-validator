# Indian Flag Image Validator

## Description
The **Indian Flag Validator** is a Python-based tool that automatically validates Indian national flags (Tiranga) for color accuracy, stripe proportions, and Ashoka Chakra specifications according to **Bureau of Indian Standards (BIS)**. 

## Context
The Indian national flag has strict rules defined by the BIS:

- **Aspect ratio:** 3:2 (width:height)  
- **Horizontal stripes:** Saffron, White, Green (each 1/3 of height)  
- **Ashoka Chakra:** 24 evenly spaced spokes, centered in the white band, diameter must be 3/4 of white band  

This project validates a given flag image (PNG, JPG, or SVG) and outputs a detailed JSON report showing compliance with BIS standards.

## Features
- Validates **aspect ratio**  
- Validates **color accuracy** (Saffron, White, Green, Chakra Blue)  
- Validates **stripe proportions**  
- Detects **Ashoka Chakra**: center position, diameter, and 24 spokes  
- Works for **any resolution** without hardcoded templates  
- Handles **local files or image URLs**  
- Supports **PNG, JPG, and SVG**  
- Processes images ≤5MB within ~3 seconds
- outputs a  JSON report with pass/fail and deviation percentages




