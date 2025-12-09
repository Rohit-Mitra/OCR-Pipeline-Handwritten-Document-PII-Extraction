from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import re
import numpy as np

class OCRPipeline:
    def __init__(self):
        self.pii_patterns = {
            'phone': r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'age': r'\b(?:age|Age|AGE)[\s:]*(\d{1,3})\b',
            'ipd_no': r'\b(?:IPD|ipd)[\s]*(?:No|no|NO)[\s.:]*([A-Z0-9]+)\b',
            'uhid': r'\b(?:UHID|uhid)[\s]*(?:No|no|NO)[\s.:]*([A-Z0-9]+)\b',
            'bed_no': r'\b(?:Bed|bed|BED)[\s]*(?:No|no|NO)[\s.:]*([A-Z0-9]+)\b',
        }
    
    def preprocess_image(self, img_path):
        img = Image.open(img_path)
        
        if img.mode != 'L':
            img = img.convert('L')
        
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        img = ImageOps.autocontrast(img)
        
        threshold = 128
        img = img.point(lambda p: p > threshold and 255)
        
        return img
    
    def deskew_image(self, img):
        np_img = np.array(img)
        
        coords = np.column_stack(np.where(np_img < 128))
        if len(coords) == 0:
            return img
        
        angle = self.calculate_skew_angle(coords)
        
        if abs(angle) > 0.5:
            img = img.rotate(angle, fillcolor=255, expand=True)
        
        return img
    
    def calculate_skew_angle(self, coords):
        if len(coords) < 2:
            return 0
        
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]
        
        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)
        
        numerator = np.sum((x_coords - mean_x) * (y_coords - mean_y))
        denominator = np.sum((x_coords - mean_x) ** 2)
        
        if denominator == 0:
            return 0
        
        angle = np.arctan(numerator / denominator) * 180 / np.pi
        
        return -angle
    
    def extract_text(self, processed_img):
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(processed_img, config=config)
        return text
    
    def clean_text(self, text):
        text = re.sub(r'[^\w\s:./\-()]', '', text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def extract_pii(self, text):
        pii_data = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pii_data[pii_type] = matches
        
        name_patterns = [
            r'(?:Patient|patient)\s*(?:Name|name|NAME)[\s:]*([A-Za-z\s]+)',
            r'(?:Name|name|NAME)[\s:]*([A-Za-z\s]+)',
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            if matches:
                names = [m.strip() for m in matches if len(m.strip()) > 2]
                if names:
                    pii_data['name'] = names
                    break
        
        sex_match = re.search(r'(?:Sex|sex|SEX)[\s:]*([MFmf])', text)
        if sex_match:
            pii_data['sex'] = [sex_match.group(1)]
        
        return pii_data
    
    def redact_image(self, img, text_data):
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        redacted = img.copy()
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(redacted)
        
        all_pii_values = []
        for values in text_data.values():
            all_pii_values.extend(values)
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text and any(pii in text for pii in all_pii_values):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                draw.rectangle([x, y, x + w, y + h], fill='black')
        
        return redacted
    
    def process(self, img_path, save_redacted=False):
        processed_img = self.preprocess_image(img_path)
        processed_img = self.deskew_image(processed_img)
        
        raw_text = self.extract_text(processed_img)
        clean_text = self.clean_text(raw_text)
        
        pii_data = self.extract_pii(clean_text)
        
        result = {
            'raw_text': raw_text,
            'clean_text': clean_text,
            'pii_data': pii_data
        }
        
        if save_redacted:
            redacted_img = self.redact_image(processed_img, pii_data)
            result['redacted_img'] = redacted_img
        
        return result

def main():
    pipeline = OCRPipeline()
    
    img_path = 'C:/Users/rohit/Documents/Projects/OCR Hand PY/sample 2.jpg'
    
    result = pipeline.process(img_path, save_redacted=True)
    
    print("Extracted Text:")
    print(result['clean_text'])
    print("\nPII Data:")
    for key, value in result['pii_data'].items():
        print(f"{key}: {value}")
    
    if 'redacted_img' in result:
        result['redacted_img'].save('redacted_output.jpg')
        print("\nRedacted image saved as 'redacted_output.jpg'")

if __name__ == "__main__":
    main()