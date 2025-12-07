# COCO Category ID to YOLO Class Index Mapping

## The Problem

COCO dataset has:
- **80 object categories**
- **Category IDs: 1-90** (non-sequential - some IDs are skipped)
- **YOLO expects: class indices 0-79** (sequential)

## What Was Wrong Before

The old code was doing:
```python
class_id = ann['category_id'] - 1  # WRONG!
```

This would map:
- category_id 1 → class 0 ✅
- category_id 2 → class 1 ✅
- category_id 80 → class 79 ✅
- category_id 81 → class 80 ❌ (exceeds max of 79!)
- category_id 82 → class 81 ❌
- category_id 83 → class 82 ❌
- etc.

**Result:** Labels with category_id 81+ were creating invalid class indices > 79.

## What the New Code Does

The new code creates a **proper mapping**:

```python
# 1. Get all categories from COCO JSON
categories = coco_data.get('categories', [])

# 2. Sort categories by their ID
sorted_categories = sorted(categories, key=lambda x: x['id'])

# 3. Map each category_id to sequential index 0-79
for idx, cat in enumerate(sorted_categories):
    category_id_to_class_idx[cat['id']] = idx
```

## Example Mapping

If COCO has categories with IDs: [1, 2, 3, ..., 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]

After sorting and mapping:
- category_id 1 → class 0
- category_id 2 → class 1
- ...
- category_id 80 → class 79 (if 80 exists)
- category_id 81 → class 80 ❌ **Wait, that's still wrong!**

## The Real Solution

Actually, COCO has exactly 80 categories, but their IDs are **not** 1-80. They're spread across 1-90 with gaps.

The mapping works like this:
1. Take all 80 categories from the JSON
2. Sort them by category_id
3. Assign sequential indices 0-79 based on their sorted position

**Example:**
If categories are: [1, 2, 3, 5, 7, ..., 81, 82, 83, 85, 87, 90]

Mapping:
- category_id 1 → class 0 (1st in sorted list)
- category_id 2 → class 1 (2nd in sorted list)
- category_id 3 → class 2 (3rd in sorted list)
- ...
- category_id 81 → class X (where X is its position in sorted list, 0-79)
- category_id 82 → class Y
- ...
- category_id 90 → class 79 (last in sorted list)

## What Happens to Category ID 81

**If category_id 81 exists in the COCO categories:**
- It gets mapped to its **position in the sorted list** (somewhere between 0-79)
- The exact index depends on how many categories with IDs < 81 exist

**If category_id 81 doesn't exist:**
- It won't be in the mapping
- Any annotations with category_id 81 will be **skipped** (with a warning)

## Verification

The code now:
1. ✅ Validates we have exactly 80 categories
2. ✅ Ensures all mapped indices are 0-79
3. ✅ Skips any annotations with category_ids not in the mapping
4. ✅ Prints the mapping range for debugging

## To See the Actual Mapping

When you run the setup script, it will print:
```
📊 Found 80 categories
📊 Category ID range: X - Y
📊 Mapped to class indices: 0 - 79
```

This shows you the actual COCO category ID range and confirms all are mapped to 0-79.

