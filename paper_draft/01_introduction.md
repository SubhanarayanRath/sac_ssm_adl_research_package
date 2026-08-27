# 1. Introduction

The ability to recognise activities of daily living inside a home has become important in assisted living and long-term health monitoring. Changes in sleeping, eating, meal preparation, bathroom use, mobility, and household routines can reveal a decline in independence or a change in health. A smart-home system that recognises these activities continuously can support caregivers without requiring constant direct observation.

Ambient sensors are well suited to this setting. Motion detectors, door contacts, and simple environmental sensors can collect useful behavioural information without recording images or speech. They also avoid the burden of wearing and charging a device. These advantages have encouraged the use of datasets such as WSU-CASAS Aruba for activity recognition research.

The problem remains difficult in a real deployment. Ambient sensors generate sparse and irregular event streams. Several events may occur within a few seconds, while long gaps may appear during sleep or periods of low movement. Activities also have very different durations. Short activities such as entering a home or moving from the bed to the toilet may be represented by only a few events. Long activities such as sleeping or relaxing can extend for hours. A fixed window therefore provides too little context for some activities and too much stale context for others.

Another difficulty comes from activity transitions. Real-time systems do not know the exact beginning and end of an activity. A sliding window can contain the end of one activity and the beginning of another. The dominant class may then suppress the recent activity. The problem becomes more serious when the dataset contains a large `Other` class and rare classes with only a small number of samples.

Existing work has addressed parts of this problem. Fixed-time windows with spatiotemporal features have produced strong results on Aruba. However, performance falls when raw events and the `Other` class are retained. Language-model-style pretraining has also improved the representation of irregular sensor sequences. Transfer between homes has shown promise, although different layouts and sensor identities still limit direct reuse. More recent work has started to study floorplan trajectories and mixed-window boundary contamination. These advances are useful, but a lightweight model that directly couples irregular time gaps with adaptive context length is still needed.

This study proposes SAC-SSM, a Semantic Adaptive-Context Selective State-Space Model for ambient-sensor ADL recognition. Each event is represented through a factorised embedding of sensor identity, sensor state, sensor type, optional room semantics, and temporal context. Three context scales are processed in parallel. A continuous-time selective state-space recurrence uses the observed inter-event gap to control memory retention. An adaptive gate then combines short, medium, and long context according to the current event pattern. Self-supervised next-event and next-gap objectives are used before activity fine-tuning. The final model is trained with a class-balanced focal objective to reduce the dominance of frequent activities.

A strict chronological protocol is used. The event stream is divided into training, validation, and test periods before overlapping windows are created. This prevents adjacent windows from different partitions from sharing most of the same events. The proposed model is compared with classical machine learning, recurrent neural networks, and the earlier CNN-Transformer-BiLSTM-Attention model. The contribution of each component is examined through ablation studies.

The main contributions are as follows:

1. A continuous-time selective state-space block is introduced for irregular ambient sensor sequences. The inter-event time gap directly controls state retention.
2. A multi-scale context design is proposed to combine short, medium, and long event histories. A learned gate selects the useful context for each prediction.
3. A factorised event representation is used to combine local sensor identity with transferable semantic information such as sensor type, room, state, and time.
4. Self-supervised next-event and next-gap prediction are integrated with supervised activity recognition.
5. A leakage-resistant chronological evaluation protocol is adopted. Macro-F1, balanced accuracy, calibration, latency, and model size are reported in addition to accuracy.
6. The model is designed for later transfer from CASAS homes to a customised real-time ADL setup through semantic sensor mapping.

The rest of the paper is organised as follows. Section 2 reviews ambient-sensor activity recognition, sequence representation, transfer learning, and deep temporal models. Section 3 presents the dataset and preprocessing procedure. Section 4 describes SAC-SSM. Section 5 explains the experiments and evaluation protocol. Section 6 discusses the results. Section 7 presents limitations and future work.
