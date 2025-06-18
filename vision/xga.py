
import cv2
import numpy as np
import pytesseract
import json

class UIElement:
    """Class to represent a UI element with a structured string representation"""
    def __init__(self, data):
        self.center = data["center"]
        self.label = data["label"]
        self.bbox = data["bbox"]
        self.width = data["width"]
        self.height = data["height"]
        self.position = data["position"]
        self.element_type = data.get("element_type", "Unknown")
        self.closest_elements = data.get("closest_elements", {})
    
    def __repr__(self):
        return f"UIElement(type='{self.element_type}', label='{self.label}', center={self.center}, size={self.width}x{self.height})"
    
    def to_dict(self):
        return {
            "center": self.center,
            "label": self.label,
            "bbox": self.bbox,
            "width": self.width,
            "height": self.height,
            "position": self.position,
            "element_type": self.element_type,
            "closest_elements": self.closest_elements
        }

class UIElementCollection:
    """Collection of UI elements with a structured representation"""
    def __init__(self, elements=None):
        self.elements = [UIElement(elem) for elem in elements] if elements else []
    
    def __repr__(self):
        if not self.elements:
            return "UIElementCollection([])"
        
        result = f"UIElementCollection with {len(self.elements)} elements:\n"
        for i, elem in enumerate(self.elements):
            result += f"  {i}: {elem}\n"
        return result
    
    def __getitem__(self, idx):
        return self.elements[idx]
    
    def __len__(self):
        return len(self.elements)
    
    def to_dict(self):
        return [elem.to_dict() for elem in self.elements]
    
    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)



def detect_grid_patterns(gray_img, ui_elements):
    """
    Detects grid-like patterns such as tic-tac-toe boards in the image
    
    Args:
        gray_img: Grayscale image
        ui_elements: Existing list of UI elements to append to
        
    Returns:
        List of UI elements with grid cells added
    """
    # Use Hough Line Transform to detect grid lines
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
    
    if lines is None:
        return ui_elements
    
    # Find horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
            horizontal_lines.append((min(x1, x2), y1, max(x1, x2), y2))
        else:  # Vertical line
            vertical_lines.append((x1, min(y1, y2), x2, max(y1, y2)))
    
    # Sort lines by position
    horizontal_lines.sort(key=lambda x: x[1])  # Sort by y-coordinate
    vertical_lines.sort(key=lambda x: x[0])    # Sort by x-coordinate
    
    # Group close lines
    h_grouped = group_close_lines(horizontal_lines, axis=1)
    v_grouped = group_close_lines(vertical_lines, axis=0)
    
    # Find grid cells by intersecting lines
    cells = []
    for i in range(len(h_grouped) - 1):
        for j in range(len(v_grouped) - 1):
            # Get average coordinates for each line
            y1 = sum(line[1] for line in h_grouped[i]) / len(h_grouped[i])
            y2 = sum(line[1] for line in h_grouped[i+1]) / len(h_grouped[i+1])
            x1 = sum(line[0] for line in v_grouped[j]) / len(v_grouped[j])
            x2 = sum(line[0] for line in v_grouped[j+1]) / len(v_grouped[j+1])
            
            # Create a cell
            cell_width = x2 - x1
            cell_height = y2 - y1
            
            # Skip very small or very large cells
            if cell_width < 20 or cell_height < 20 or cell_width > 200 or cell_height > 200:
                continue
                
            # Add to cells if it's square-like (aspect ratio close to 1)
            aspect_ratio = cell_width / cell_height if cell_height > 0 else 0
            if 0.7 <= aspect_ratio <= 1.3:
                cells.append((int(x1), int(y1), int(cell_width), int(cell_height)))
    
    # Add grid cells to UI elements
    for i, (x, y, w, h) in enumerate(cells):
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check if the cell overlaps with existing elements
        is_duplicate = False
        for elem in ui_elements:
            ex, ey, ew, eh = elem["bbox"]
            overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
            overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
            overlap_area = overlap_x * overlap_y
            min_area = min(w * h, ew * eh)
            
            if min_area > 0 and overlap_area / min_area > 0.5:
                is_duplicate = True
                break
        
        if not is_duplicate:
            # Add grid cell as a UI element
            ui_elements.append({
                "center": (center_x, center_y),
                "label": f"Grid Cell ({center_x}, {center_y})",
                "bbox": (x, y, w, h),
                "width": w,
                "height": h,
                "position": (x, y),
                "element_type": "Grid Cell"
            })
    
    return ui_elements

def group_close_lines(lines, axis=0, threshold=15):
    """
    Group lines that are close to each other
    
    Args:
        lines: List of lines (x1, y1, x2, y2)
        axis: 0 for vertical lines (group by x), 1 for horizontal lines (group by y)
        threshold: Maximum distance between lines to be grouped
        
    Returns:
        List of grouped lines
    """
    if not lines:
        return []
    
    # Sort lines by the specified axis
    sorted_lines = sorted(lines, key=lambda x: x[axis])
    
    # Group lines
    groups = [[sorted_lines[0]]]
    for line in sorted_lines[1:]:
        # Get the last group
        last_group = groups[-1]
        last_coord = sum(line[axis] for line in last_group) / len(last_group)
        
        # Check if current line is close to the last group
        if line[axis] - last_coord < threshold:
            # Add to existing group
            last_group.append(line)
        else:
            # Create new group
            groups.append([line])
    
    return groups


def detect_ui_elements_from_image(image):
    """
    Detects UI elements in an image and returns a structured representation.
    Enhanced to detect clickable text and rounded buttons.
    
    Args:
        image: Can be either a numpy array (cv2 image) or a path to an image file
        
    Returns:
        UIElementCollection: Collection of UI elements with structured representation
    """
    tesseract_path = r'Tesseract-OCR\tesseract.exe'
    # Configure Tesseract path if provided
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Handle different input types
    if isinstance(image, str):
        # Image path provided
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not open or find the image: {image}")
    else:
        # Assume numpy array
        img = image.copy()
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # List to store detected UI elements
    ui_elements = []
    
    # 1. Detect text elements first to identify clickable text
    # Use better configuration for text detection
    custom_config = r'--oem 3 --psm 11'  # Use advanced OCR Engine Mode and full page segmentation
    
    # First, detect all text blocks (not individual words)
    # This helps prevent detecting single letters as separate elements
    blocks = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    # Group text by block_num and par_num to get proper text blocks
    text_blocks = {}
    for i in range(len(blocks['text'])):
        if blocks['text'][i].strip():  # Skip empty text
            block_key = f"{blocks['block_num'][i]}-{blocks['par_num'][i]}"
            
            if block_key not in text_blocks:
                text_blocks[block_key] = {
                    "text": [],
                    "left": [],
                    "top": [],
                    "width": [],
                    "height": []
                }
            
            text_blocks[block_key]["text"].append(blocks['text'][i])
            text_blocks[block_key]["left"].append(blocks['left'][i])
            text_blocks[block_key]["top"].append(blocks['top'][i])
            text_blocks[block_key]["width"].append(blocks['width'][i])
            text_blocks[block_key]["height"].append(blocks['height'][i])
    
    # Further merge adjacent text blocks that belong together
    processed_blocks = []
    
    # Convert text_blocks from dict to list
    block_list = list(text_blocks.values())
    
    # Process each text block
    for block in block_list:
        if not block["text"]:
            continue
            
        # Calculate the bounding box for the text block
        x_min = min(block["left"])
        y_min = min(block["top"])
        x_max = max([block["left"][i] + block["width"][i] for i in range(len(block["left"]))])
        y_max = max([block["top"][i] + block["height"][i] for i in range(len(block["top"]))])
        
        # Combine the text
        text = " ".join(block["text"])
        
        # Create a processed block
        processed_block = {
            "text": text,
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min
        }
        
        # Try to merge with existing blocks if they're very close
        merged = False
        for existing_block in processed_blocks:
            # Calculate distance between blocks
            dist_x = min(
                abs(existing_block["x"] + existing_block["width"] - processed_block["x"]),
                abs(processed_block["x"] + processed_block["width"] - existing_block["x"])
            )
            dist_y = min(
                abs(existing_block["y"] + existing_block["height"] - processed_block["y"]),
                abs(processed_block["y"] + processed_block["height"] - existing_block["y"])
            )
            
            # Check if they're on the same line and close horizontally
            same_line = abs(existing_block["y"] - processed_block["y"]) < max(existing_block["height"], processed_block["height"]) * 0.5
            close_horizontally = dist_x < max(existing_block["height"], processed_block["height"])
            
            if same_line and close_horizontally:
                # Merge blocks
                x_min = min(existing_block["x"], processed_block["x"])
                y_min = min(existing_block["y"], processed_block["y"])
                x_max = max(existing_block["x"] + existing_block["width"], processed_block["x"] + processed_block["width"])
                y_max = max(existing_block["y"] + existing_block["height"], processed_block["y"] + processed_block["height"])
                
                # Update existing block
                existing_block["text"] = existing_block["text"] + " " + processed_block["text"]
                existing_block["x"] = x_min
                existing_block["y"] = y_min
                existing_block["width"] = x_max - x_min
                existing_block["height"] = y_max - y_min
                
                merged = True
                break
        
        if not merged:
            processed_blocks.append(processed_block)
    
    # Create UI elements from the processed text blocks
    for block in processed_blocks:
        # Skip very small text (likely noise)
        if block["width"] < 15 or block["height"] < 10:
            continue
            
        # Skip very large text blocks (likely paragraphs, not clickable)
        if block["width"] > img.shape[1] / 2 and block["height"] > 50:
            continue
            
        # Calculate center position
        center_x = block["x"] + block["width"] // 2
        center_y = block["y"] + block["height"] // 2
        
        # Heuristic to identify clickable text
        is_clickable = False
        
        # Length-based heuristic: clickable text tends to be short
        if len(block["text"]) < 30:
            is_clickable = True
        
        # Check if the text appears to be a link/button
        clickable_keywords = ['login', 'sign', 'submit', 'click', 'register', 'buy', 
                             'download', 'upload', 'send', 'save', 'cancel', 'ok', 
                             'yes', 'no', 'next', 'back', 'previous', 'continue', 'for you',
                             'home', 'menu', 'search', 'profile', 'settings', 'logout']
        
        for keyword in clickable_keywords:
            if keyword in block["text"].lower():
                is_clickable = True
                break
        
        # Check if text is short enough to be a navigation item
        if len(block["text"].split()) <= 3:
            is_clickable = True
        
        if is_clickable:
            ui_elements.append({
                "center": (center_x, center_y),
                "label": block["text"],
                "bbox": (block["x"], block["y"], block["width"], block["height"]),
                "width": block["width"],
                "height": block["height"],
                "position": (block["x"], block["y"]),
                "element_type": "Clickable Text"
            })
    
    # 2. Detect rectangular and rounded buttons
    # Apply adaptive thresholding to handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Add morphological operations to better detect rounded corners
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to identify potential UI elements
    for contour in contours:
        # Get bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size to remove noise (adjust thresholds based on your needs)
        if w > 30 and h > 15:
            # Calculate center position
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Extract ROI for text recognition and shape analysis
            roi = gray[y:y+h, x:x+w]
            
            # Check if this is a rounded button by analyzing contour
            # Calculate contour approximation
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Determine if this is likely a rounded button
            is_rounded = False
            if len(approx) > 4 and len(approx) < 15:  # More points than rectangle but not too complex
                is_rounded = True
            
            # Calculate contour solidity (area / convex hull area)
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                # Rounded buttons often have high solidity but not quite 1.0
                if 0.85 <= solidity < 0.98:
                    is_rounded = True
            
            # Use pytesseract to extract text
            try:
                # Improve text recognition by preprocessing
                roi_processed = cv2.GaussianBlur(roi, (3, 3), 0)
                roi_processed = cv2.threshold(roi_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Extract text
                text = pytesseract.image_to_string(roi_processed).strip()
                
                # If no text is found, try to provide a generic element name
                if not text:
                    # Determine element type based on shape properties
                    aspect_ratio = w / h
                    
                    if is_rounded:
                        element_type = "Rounded Button"
                    elif 0.9 <= aspect_ratio <= 1.1 and w >= 20:
                        element_type = "Square Button"
                    elif aspect_ratio > 3:
                        element_type = "Input Field"
                    else:
                        element_type = "UI Element"
                    
                    text = f"{element_type} at ({center_x}, {center_y})"
                else:
                    # If text is found, it's likely a button or input field
                    if is_rounded:
                        element_type = "Rounded Button"
                    else:
                        element_type = "Button"
                
                # Skip this element if it overlaps significantly with a clickable text
                should_skip = False
                for elem in ui_elements:
                    if elem.get("element_type") == "Clickable Text":
                        ex, ey, ew, eh = elem["bbox"]
                        # Calculate overlap
                        overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                        overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                        overlap_area = overlap_x * overlap_y
                        min_area = min(w * h, ew * eh)
                        
                        if min_area > 0 and overlap_area / min_area > 0.7:
                            should_skip = True
                            break
                
                if not should_skip:
                    # Add to our list of UI elements
                    ui_elements.append({
                        "center": (center_x, center_y),
                        "label": text,
                        "bbox": (x, y, w, h),
                        "width": w,
                        "height": h,
                        "position": (x, y),
                        "element_type": element_type if 'element_type' in locals() else "UI Element"
                    })
            except Exception as e:
                # Just skip this element if there's an error
                continue
    
    # 3. Detect icons and other non-rectangular elements
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get minimum enclosing circle for non-rectangular shapes
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(contour)
        center_x, center_y = int(center_x), int(center_y)
        radius = int(radius)
        
        # Filter by size
        if 10 < radius < 50:
            # Check if this element overlaps with already detected elements
            is_duplicate = False
            for elem in ui_elements:
                existing_center = elem["center"]
                distance = np.sqrt((center_x - existing_center[0])**2 + (center_y - existing_center[1])**2)
                if distance < 20:  # Threshold for considering as duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                x = center_x - radius
                y = center_y - radius
                w = 2 * radius
                h = 2 * radius
                ui_elements.append({
                    "center": (center_x, center_y),
                    "label": f"Icon at ({center_x}, {center_y})",
                    "bbox": (x, y, w, h),
                    "width": w,
                    "height": h,
                    "position": (x, y),
                    "element_type": "Icon"
                })
    
    # 4. Detect grid patterns (like tic-tac-toe)
    ui_elements = detect_grid_patterns(gray, ui_elements)
    
    # Compute spatial relationships between elements
    for i, element in enumerate(ui_elements):
        center_x, center_y = element["center"]
        
        # Initialize closest elements in each direction
        closest = {
            "top": {"element": None, "distance": float('inf')},
            "bottom": {"element": None, "distance": float('inf')},
            "left": {"element": None, "distance": float('inf')},
            "right": {"element": None, "distance": float('inf')}
        }
        
        # Compare with all other elements
        for j, other in enumerate(ui_elements):
            if i == j:  # Skip self
                continue
                
            other_center_x, other_center_y = other["center"]
            
            # Calculate the distance between centers
            distance = np.sqrt((center_x - other_center_x)**2 + (center_y - other_center_y)**2)
            
            # Determine the direction
            dx = other_center_x - center_x
            dy = other_center_y - center_y
            
            # Improved direction determination with angle-based approach
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            # Assign to directions based on angle
            # Top: -45° to 45°
            if -45 <= angle < 45:
                if distance < closest["right"]["distance"]:
                    closest["right"]["element"] = j
                    closest["right"]["distance"] = distance
            # Bottom: 45° to 135°
            elif 45 <= angle < 135:
                if distance < closest["bottom"]["distance"]:
                    closest["bottom"]["element"] = j
                    closest["bottom"]["distance"] = distance
            # Left: 135° to -135°
            elif 135 <= angle or angle < -135:
                if distance < closest["left"]["distance"]:
                    closest["left"]["element"] = j
                    closest["left"]["distance"] = distance
            # Top: -135° to -45°
            elif -135 <= angle < -45:
                if distance < closest["top"]["distance"]:
                    closest["top"]["element"] = j
                    closest["top"]["distance"] = distance
        
        # Add relationship information to the element
        element["closest_elements"] = {}
        for direction, info in closest.items():
            if info["element"] is not None:
                related_element = ui_elements[info["element"]]
                element["closest_elements"][direction] = {
                    "index": info["element"],
                    "label": related_element["label"],
                    "distance": round(info["distance"], 2),
                    "width": related_element["width"],
                    "height": related_element["height"]
                }
            else:
                # Ensure we always have all four directions, even if empty
                element["closest_elements"][direction] = None
    
    return UIElementCollection(ui_elements)

def visualize_ui_elements(image, elements, output_path=None):
    """
    Visualizes detected UI elements with green borders and index numbers as tickets.

    Args:
        image: Can be either a numpy array (cv2 image) or a path to an image file
        elements: Collection of UI elements (UIElementCollection or list of dictionaries)
        output_path: Optional path to save the visualized image
        
    Returns:
        numpy.ndarray: Image with UI elements visualized
    """
    # Convert UIElementCollection to list of dictionaries if needed
    if isinstance(elements, UIElementCollection):
        elements = elements.to_dict()
    
    # Handle different input types
    if isinstance(image, str):
        # Image path provided
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not open or find the image: {image}")
    else:
        # Assume numpy array
        img = image.copy()
    
    # Draw borders and index numbers as tickets
    for i, element in enumerate(elements):
        # Get bounding box
        x, y, w, h = element["bbox"]
        element_type = element.get("element_type", "Unknown")
        
        # Use different colors based on element type
        if "Button" in element_type:
            color = (0, 255, 0)  # Green for buttons
        elif "Text" in element_type:
            color = (0, 165, 255)  # Orange for clickable text
        elif "Icon" in element_type:
            color = (255, 0, 0)  # Blue for icons
        elif "Input" in element_type:
            color = (255, 0, 255)  # Purple for input fields
        else:
            color = (0, 255, 255)  # Yellow for other elements
            
        # Draw border around the UI element
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Draw ticket above the element
        ticket_width = 25
        ticket_height = 20
        ticket_x = x + w // 2 - ticket_width // 2
        ticket_y = y - ticket_height - 5  # Position above the element
        
        # Ensure ticket stays within image bounds
        if ticket_y < 0:
            # If not enough space above, put it below
            ticket_y = y + h + 5
        
        # Create ticket background (filled rectangle)
        cv2.rectangle(img, 
                     (ticket_x, ticket_y), 
                     (ticket_x + ticket_width, ticket_y + ticket_height), 
                     color, 
                     -1)  # -1 fills the rectangle
        
        # Add border to ticket
        cv2.rectangle(img, 
                     (ticket_x, ticket_y), 
                     (ticket_x + ticket_width, ticket_y + ticket_height), 
                     (0, 0, 0), 
                     1)  # 1 pixel border
        
        # Add index number to ticket
        text_size = cv2.getTextSize(f"{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = ticket_x + (ticket_width - text_size[0]) // 2
        text_y = ticket_y + (ticket_height + text_size[1]) // 2
        cv2.putText(img, 
                   f"{i}", 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (0, 0, 0),  # Black text
                   2)
    
    # Save the visualized image if output path is provided
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img

# Example usage
# if __name__ == "__main__":
#     # Example with an image path
#     image_path = r"1.png"  # Replace with your image path
    
#     try:
#         # Detect UI elements
#         elements = detect_ui_elements_from_image(image_path)
        
#         # Print structured output with element type, width, height, and closest elements
#         print(elements)
        
#         # For detailed JSON output:
#         # print(elements.to_json())
        
#         # Visualize elements with ticket-style numbering
#         visualize_ui_elements(image_path, elements, "ui_elements_visualization.png")
        
#     except Exception as e:
#         print(f"Error: {e}")