# 🌊 SKEMA Validation Coordinates - Known Kelp Farm Locations

**Date**: January 9, 2025  
**Purpose**: Specific coordinates for validating our SKEMA integration against known kelp farms  
**Source**: Research literature + validation data framework

---

## 🎯 **British Columbia Validation Sites**

### **Primary Validation Locations** 🇨🇦

#### **1. Broughton Archipelago** (UVic Research Site)
- **Coordinates**: `50.0833°N, 126.1667°W`
- **Location**: Queen Charlotte Strait, BC
- **Source**: UVic SKEMA validation studies
- **Expected Kelp**: *Nereocystis luetkeana* (Bull kelp)
- **Validation Season**: June-September (peak kelp growth)
- **Research Notes**: Primary site used in SKEMA temporal validation studies

#### **2. Saanich Inlet Kelp Forests** (Victoria Access)
- **Coordinates**: `48.5830°N, 123.5000°W`
- **Location**: Saanich Inlet, Vancouver Island, BC
- **Source**: Validation Data Framework (Task 2.1)
- **Expected Kelp**: *Nereocystis luetkeana*, *Macrocystis pyrifera*
- **Validation Season**: July-October
- **Research Notes**: Sheltered waters, easy access, diverse depth zones

#### **3. Haro Strait (Gulf Islands)** (Ferry Access)
- **Coordinates**: `48.5000°N, 123.1667°W`
- **Location**: Between Vancouver Island and San Juan Islands
- **Source**: Validation Data Framework (Task 2.1)  
- **Expected Kelp**: Dense *Nereocystis* beds
- **Validation Season**: June-September
- **Research Notes**: Clear water, minimal human impact

#### **4. Tofino/Ucluelet Kelp Forests** (West Coast)
- **Coordinates**: `49.1667°N, 125.9167°W`
- **Location**: West Coast Vancouver Island
- **Source**: Validation Data Framework (Task 2.1)
- **Expected Kelp**: Extensive *Nereocystis* forests
- **Validation Season**: May-October
- **Research Notes**: Pristine conditions, large kelp areas, ocean-facing

---

## 🌍 **International Validation Sites**

### **California Locations** 🇺🇸

#### **5. Monterey Bay Kelp Forest**
- **Coordinates**: `36.8000°N, 121.9000°W`
- **Location**: Monterey Bay, California
- **Source**: Test integration data
- **Expected Kelp**: *Macrocystis pyrifera* (Giant kelp)
- **Validation Season**: Year-round (peak April-October)
- **Research Notes**: Well-studied kelp ecosystem, extensive research data available

#### **6. Point Reyes Kelp Beds**
- **Coordinates**: `38.0500°N, 122.9500°W`
- **Location**: Northern California coast
- **Source**: California kelp mapping studies
- **Expected Kelp**: *Nereocystis luetkeana*, *Macrocystis pyrifera*
- **Validation Season**: April-November
- **Research Notes**: Mixed kelp species, good for multi-species validation

### **Southern Hemisphere Locations** 🇦🇺

#### **7. Tasmania Kelp Forests**
- **Coordinates**: `43.1000°S, 147.3000°E`
- **Location**: Tasmania, Australia
- **Source**: Test integration data
- **Expected Kelp**: *Macrocystis pyrifera*
- **Validation Season**: October-April (Southern Hemisphere summer)
- **Research Notes**: Cool-water kelp forests, different environmental conditions

---

## 🚫 **Control Sites (No Kelp Expected)**

### **8. Mojave Desert Control**
- **Coordinates**: `36.0000°N, 118.0000°W`
- **Location**: Mojave Desert, California (inland)
- **Expected Kelp**: None (land)
- **Purpose**: Negative control for kelp detection algorithms

### **9. Open Ocean Control**
- **Coordinates**: `45.0000°N, 135.0000°W`
- **Location**: North Pacific Ocean (deep water)
- **Expected Kelp**: None (too deep)
- **Purpose**: Negative control for ocean areas without kelp

---

## 🧪 **Validation Test Framework**

### **High Priority Testing Coordinates**
For immediate SKEMA validation (Task A2.4-A2.6):

```python
SKEMA_VALIDATION_COORDINATES = {
    "high_priority": [
        {"name": "Broughton_Archipelago", "lat": 50.0833, "lng": -126.1667, "expected_kelp": True},
        {"name": "Saanich_Inlet", "lat": 48.5830, "lng": -123.5000, "expected_kelp": True},
        {"name": "Haro_Strait", "lat": 48.5000, "lng": -123.1667, "expected_kelp": True},
        {"name": "Monterey_Bay", "lat": 36.8000, "lng": -121.9000, "expected_kelp": True},
    ],
    "control_sites": [
        {"name": "Mojave_Desert", "lat": 36.0000, "lng": -118.0000, "expected_kelp": False},
        {"name": "Open_Ocean", "lat": 45.0000, "lng": -135.0000, "expected_kelp": False},
    ]
}
```

### **Validation Success Criteria**
- **True Positive Rate**: >85% detection at known kelp farm locations
- **False Positive Rate**: <15% detection at control sites (no kelp)
- **Spatial Accuracy**: Kelp extent correlation >80% with SKEMA ground truth
- **Temporal Consistency**: Detection persistence across multiple dates

### **Testing Schedule**
1. **Phase 1**: Test current system against these coordinates (baseline)
2. **Phase 2**: Implement SKEMA formulas and re-test (improvement validation)
3. **Phase 3**: Optimize thresholds based on validation results
4. **Phase 4**: Comprehensive testing across all seasons/conditions

---

## 📊 **Expected Results by Location**

| Location | Expected Detection Rate | Kelp Species | Best Months | Notes |
|----------|------------------------|--------------|-------------|--------|
| Broughton Archipelago | 90%+ | *Nereocystis* | Jul-Sep | SKEMA reference site |
| Saanich Inlet | 85%+ | Mixed | Jul-Oct | Sheltered, consistent |
| Haro Strait | 80%+ | *Nereocystis* | Jun-Sep | Clear water |
| Tofino/Ucluelet | 85%+ | *Nereocystis* | May-Oct | Large forests |
| Monterey Bay | 85%+ | *Macrocystis* | Apr-Oct | Year-round presence |
| Tasmania | 80%+ | *Macrocystis* | Oct-Apr | Different hemisphere |
| Control Sites | <5% | None | Any | Should be near-zero |

---

## 🔗 **Implementation Integration**

### **Code Integration Points**
- Add coordinates to `tests/integration/test_skema_validation.py`
- Create validation framework in `src/kelpie_carbon_v1/validation/skema_validation.py`
- Update API endpoints to include validation testing mode
- Add automated validation to CI/CD pipeline

### **Documentation Updates**
- Link to SKEMA research documents
- Reference validation data framework
- Update task lists with specific coordinate testing
- Create implementation summaries for each validation phase

---

**Last Updated**: January 9, 2025  
**Next Review**: After SKEMA formula implementation (Task A2.1-A2.3)  
**Purpose**: Enable systematic validation of SKEMA integration against scientifically validated kelp farm locations 