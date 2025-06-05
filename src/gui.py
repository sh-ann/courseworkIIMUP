import tkinter as tk
from tkinter import filedialog
import os
from src.utils import MidiPreprocessor
from src.preprocessing import filter_data
import torch


def create_gui():
    root = tk.Tk()
    root.title("MIDI Processor")
    root.geometry("500x300")
    root.configure(padx=20, pady=20)

    # Элементы интерфейса
    header = tk.Label(root, text="MIDI Processor", font=("Arial", 16, "bold"))
    header.pack(pady=10)

    instructions = tk.Label(
        root, text="Select a MIDI file to process and convert to MusicXML."
    )
    instructions.pack(pady=5)

    process_btn = tk.Button(root, text="Process MIDI File", command=process_midi_file)
    process_btn.pack(pady=20)

    status_label = tk.Label(root, text="Ready", fg="green")
    status_label.pack(pady=10)

    root.mainloop()


def process_midi():
    file_path = filedialog.askopenfilename(
        title="Select MIDI File",
        filetypes=[("MIDI files", "*.mid;*.midi"), ("All files", "*.*")],
    )
    if not file_path:
        return

    try:
        midi_data = mido.MidiFile(file_path)
        print(f"Processed: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")


def process_midi_file():
    os.makedirs("./resultMIDI", exist_ok=True)
    os.makedirs("./resultXML", exist_ok=True)

    midi_file_path = filedialog.askopenfilename(
        title="Select MIDI File to Process",
        filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")],
    )

    if not midi_file_path:
        print("No file selected. Operation cancelled.")
        return

    midi_preprocessor = MidiPreprocessor()

    processed_midi = midi_preprocessor.parse(midi_file_path)

    model = torch.load(
        "/Users/georgij/Desktop/Универ/помощь/Anya/midi_project_ABSOLUTELY_CLEAN_OK/models/final_model.pt",
        weights_only=False,
    )

    clean_midi = filter_data(processed_midi, model)

    output_midi_path = os.path.join("./resultMIDI", os.path.basename(midi_file_path))
    output_xml_path = os.path.join(
        "./resultXML", os.path.splitext(os.path.basename(midi_file_path))[0] + ".xml"
    )

    xml_path = convert_midi_to_musicxml(clean_midi, output_midi_path, output_xml_path)

    result_label.config(text=f"Processing complete!\nMusicXML saved to: {xml_path}")

    # open_folder_button.config(state=tk.NORMAL)
    # global current_output_folder
    # current_output_folder = os.path.dirname(xml_path)
