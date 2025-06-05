import os
import numpy as np


class NoteEvent:
    def __init__(self, note_on_msg, note_off_msg, start_time, end_time):
        self.note_on_msg = note_on_msg
        self.note_off_msg = note_off_msg
        self.start_time = start_time
        self.end_time = end_time

    def duration(self):
        return self.end_time - self.start_time

    def quantized_start(self, q):
        return round(self.start_time / q) * q

    def quantized_duration(self, q):
        return round(self.duration() / q) * q


def compute_absolute_times(track):
    total_time = 0
    for msg in track:
        total_time += msg.time
        yield msg, total_time


def build_time_dict(events):
    return {(msg.note, msg.velocity): value for msg, value in events}


def matching_note_search(note_off_msg, start_times):
    new_start_times = []
    matched_pair = None

    for note_on_msg, start_time in start_times:
        if (
            note_on_msg.note == note_off_msg.note
            and note_on_msg.channel == note_off_msg.channel
            and matched_pair is None
        ):
            duration = note_off_msg.time - start_time
            matched_pair = (note_on_msg, duration)
        else:
            new_start_times.append((note_on_msg, start_time))

    return new_start_times, matched_pair


def recalculate_note_off_time(
    matching_notes, current_time, start_times_dict, durations_dict
):
    for note_on in matching_notes:
        if isinstance(note_on, mido.Message):
            key = (note_on.note, note_on.velocity)
            if key in start_times_dict and key in durations_dict:
                return start_times_dict[key] + durations_dict[key] - current_time
    return 0


def process_track(track, start_times_dict, durations_dict, start_times):
    updated_msgs = []
    absolute_times = list(compute_absolute_times(track))

    for msg, current_time in absolute_times:
        if msg.type not in ["note_on", "note_off"]:
            updated_msgs.append(msg)
            continue

        updated_msg = msg.copy()
        note_key = (msg.note, msg.velocity)

        if msg.type == "note_on" and msg.velocity > 0:
            if note_key in start_times_dict:
                updated_msg.time = start_times_dict[note_key] - current_time

        elif msg.type in ["note_off", "note_on"] and msg.velocity == 0:
            matching_notes, _ = matching_note_search(msg, start_times)
            updated_msg.time = recalculate_note_off_time(
                matching_notes, current_time, start_times_dict, durations_dict
            )

        updated_msgs.append(updated_msg)

    return updated_msgs


def reconstruct_midi_data(midi_data, start_times, durations):
    start_times_dict = build_time_dict(start_times)
    durations_dict = build_time_dict(durations)

    for track in midi_data.tracks:
        processed = process_track(track, start_times_dict, durations_dict, start_times)
        track[:] = processed

    return midi_data


def quantize_note_timings(midi_data, quantize_to=16, mode="both"):
    ticks_per_quantize = midi_data.ticks_per_beat // quantize_to
    note_groups = {}
    all_events = []

    for track in midi_data.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_groups.setdefault((msg.note, msg.channel), []).append(
                    (msg, abs_time)
                )
            elif msg.type in ["note_off", "note_on"] and msg.velocity == 0:
                key = (msg.note, msg.channel)
                if key in note_groups and note_groups[key]:
                    note_on_msg, start_time = note_groups[key].pop(0)
                    all_events.append(NoteEvent(note_on_msg, msg, start_time, abs_time))

    start_times = []
    durations = []

    for event in all_events:
        q_start = (
            event.quantized_start(ticks_per_quantize)
            if mode in ["start_times", "both"]
            else event.start_time
        )
        q_dur = (
            event.quantized_duration(ticks_per_quantize)
            if mode in ["durations", "both"]
            else event.duration()
        )
        start_times.append((event.note_on_msg, q_start))
        durations.append((event.note_off_msg, q_dur))

    return reconstruct_midi_data(midi_data, start_times, durations)


def preprocess_data(midi_data, sequence_length=32):
    midi_events = []

    for track in midi_data.tracks:
        for event in track:
            if hasattr(event, "note"):
                midi_events.append([event.note, event.time])
            else:
                midi_events.append([0, event.time])

    sequences = []
    for i in range(len(midi_events) - sequence_length):
        sequences.append(midi_events[i : i + sequence_length])

    return np.array(sequences)


def load_midi_files(file_directory):
    midi_files = []
    for root, dirs, files in os.walk(file_directory):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                midi_files.append(os.path.join(root, file))

    return midi_files


def filter_data(midi_data, model):
    input_notes = preprocess_data(midi_data)
    note_predictions = model(input_notes)

    clean_midi_data = postprocess_predictions(note_predictions, midi_data)
    return clean_midi_data


def postprocess_predictions(
    predictions, midi_data, threshold=0.000005, output_file_path="output.mid"
):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    events_with_predictions = [
        (event, predictions[i])
        for track in midi_data.tracks
        for i, event in enumerate(track)
        if hasattr(event, "note")
    ]

    for event, prediction in events_with_predictions:
        if prediction >= threshold and event.time >= 0:
            track.append(event)
            if event.type == "note_on":
                note_off_event = mido.Message(
                    "note_off", note=event.note, velocity=64, time=event.time
                )
                track.append(note_off_event)

    mid.save(output_file_path)
    return mid
