# Lost & Unattended Object Detection System  
**Applied Computer Vision | Real-Time Analytics**

## Overview
This project implements a **real-time computer vision system** to detect, track, and flag **unattended personal items** in video streams, supporting Lost & Found and operational monitoring use cases.  
The focus is on **decision logic, reliability, and privacy-by-design**, rather than raw model accuracy alone.

The system demonstrates how computer vision can be operationalized into **actionable alerts** by combining object detection, multi-object tracking, and rule-based association logic.

---

## Problem Statement
In high-traffic indoor environments, distinguishing **temporarily placed items** from **truly lost objects** is non-trivial.  
Naïve object detection systems generate excessive false positives, making them impractical for real-world operations.

This system addresses that gap by translating an ambiguous business problem into **clear detection rules, thresholds, and alerting logic** that stakeholders can understand and trust.

---

## System Architecture
**Pipeline Overview**
1. Video stream ingestion  
2. Object detection (people + items)  
3. Multi-object tracking across frames  
4. Owner–item association logic  
5. Unattended item classification  
6. Alert generation with privacy safeguards  

**Core Components**
- **Detection:** YOLOv8 for real-time object detection  
- **Tracking:** ByteTrack for persistent tracking of people and objects  
- **Association Logic:** Spatial proximity + dwell-time heuristics  
- **Decision Layer:** Threshold-based unattended item classification  
- **Output:** Snapshot-based alerts with metadata logging  
- **Privacy Controls:** Face blurring and metadata-only storage  

---

## Key Features
- Real-time detection and tracking of personal items and people  
- Owner–item association using spatial and temporal heuristics  
- Tunable thresholds to reduce false positives in crowded scenes  
- Automated alert generation with visual evidence  
- Privacy-first design (face blurring, no raw video storage)  

---

## Approach
1. Detect people and candidate objects per frame using YOLOv8  
2. Track entities across frames with ByteTrack  
3. Associate objects to nearest owners using distance and motion continuity  
4. Monitor dwell time and inactivity windows to identify unattended items  
5. Trigger alerts once business-defined thresholds are exceeded  
6. Apply privacy filters before logging or alerting  

---

## Validation
- Evaluated on **campus-style indoor video scenarios**  
- Iteratively tuned detection confidence, dwell-time, and proximity thresholds  
- Performance assessed based on **false-positive reduction and alert relevance**, not just detection accuracy  

---

## Business Value
- **Actionable insights:** Converts raw video into discrete Lost & Found alerts instead of continuous monitoring  
- **Reduced alert fatigue:** Owner–item association logic significantly lowers false positives  
- **Operational efficiency:** Snapshot-based alerts reduce manual review time  
- **Compliance-ready:** Privacy safeguards align with enterprise data protection expectations  
- **Configurable decision layer:** Thresholds can be adapted without retraining models  

---

## Trade-offs & Design Decisions
- **Rules vs. pure ML:** Chose deterministic, rule-based logic over end-to-end learning to improve explainability and stakeholder trust  
- **Precision over recall:** Prioritized fewer, higher-confidence alerts at the cost of delayed or missed detections in ambiguous cases  
- **Metadata over raw video:** Avoided storing full video to reduce privacy risk, trading off post-event forensic depth  
- **Real-time constraints:** Balanced accuracy against latency and compute cost  

---

## Limitations
- Rule-based association may struggle in **extremely dense crowds** with frequent occlusions  
- Static camera assumptions limit robustness to dynamic camera movement  
- Thresholds require **environment-specific tuning** for optimal performance  
- System is not designed for long-term forensic investigation due to limited data retention  

---

## Future Enhancements
- Incorporate **re-identification embeddings** to improve owner–item linkage in crowded scenes  
- Adaptive thresholding based on scene density and time-of-day patterns  
- Integration with notification systems (email, dashboard, ticketing tools)  
- Extension to additional object categories and outdoor environments  
- Lightweight model optimization for edge deployment  

---

## Tech Stack
- Python  
- YOLOv8  
- ByteTrack  
- OpenCV  
- NumPy  

---

## Use Cases
- Campus Lost & Found monitoring  
- Retail and public facility surveillance support  
- Operational safety and asset monitoring  
- Smart building analytics  

---

## Disclaimer
This repository is intended as a **reference implementation** demonstrating applied system design and analytics trade-offs.  
It is not a production surveillance product.
