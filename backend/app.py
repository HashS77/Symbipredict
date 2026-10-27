#Import necessary libraries
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Legacy feature list from the previous model; runtime features come from the dataset below.
_previous_model_features = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", "spotting_ urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic _patches", "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload.1", "blood_in_sputum", "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]
label_column = 'prognosis'  # The target column in the dataset is specified here.
# What: name of the target (label) column in the CSV.
# Why: used to encode labels and train the classifier.
# What it does: LabelEncoder maps textual disease names to numeric classes.

# Train and save model if not present
# What: check for existing model to avoid retraining every run.
# Why: training is expensive; load saved model if available for faster startup.
# What it does: if model.h5 missing -> load CSV, train model, save model + label encoder; otherwise load both.
label_column = "diseases"
backend_dir = os.path.dirname(__file__)
dataset_path = os.path.join(backend_dir, "dataset", "symbipredict_2022.csv")
model_dir = os.path.join(backend_dir, "models")
model_path = os.path.join(model_dir, "model.h5")
encoder_path = os.path.join(model_dir, "label_encoder.pkl")
metadata_path = os.path.join(model_dir, "model_metadata.json")
dataset_columns = pd.read_csv(dataset_path, nrows=0).columns.tolist()
if label_column not in dataset_columns:
    raise ValueError(f"Dataset must include a {label_column!r} label column.")
feature_columns = [column for column in dataset_columns if column != label_column]
data = pd.read_csv(dataset_path, dtype={column: np.uint8 for column in feature_columns})
if data[label_column].isna().any():
    raise ValueError(f"Dataset contains missing values in the {label_column!r} label column.")
if not np.isin(data[feature_columns].to_numpy(), (0, 1)).all():
    raise ValueError("Dataset symptom features must contain only 0/1 values.")

labels = data[label_column].astype(str)
expected_labels = sorted(labels.unique().tolist())
training_seed = 42
expected_metadata = {
    "features": feature_columns,
    "labels": expected_labels,
    "training_seed": training_seed,
}
model = None
le = None
if all(os.path.exists(path) for path in (model_path, encoder_path, metadata_path)):
    with open(metadata_path, encoding="utf-8") as metadata_file:
        saved_metadata = json.load(metadata_file)
    if saved_metadata == expected_metadata:
        model = tf.keras.models.load_model(model_path)
        with open(encoder_path, "rb") as encoder_file:
            le = pickle.load(encoder_file)
        if (
            model.input_shape[-1] != len(feature_columns)
            or model.output_shape[-1] != len(expected_labels)
            or le.classes_.tolist() != expected_labels
        ):
            model = None
            le = None

if model is None or le is None:
    print("Saved model is missing or out of date; training it against the current dataset.")
    os.makedirs(model_dir, exist_ok=True)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = data[feature_columns].to_numpy(dtype=np.float32)
    tf.keras.utils.set_random_seed(training_seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(feature_columns),)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(le.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=512,
        validation_data=(X_test, y_test),
        verbose=2,
    )
    model.save(model_path)
    with open(encoder_path, "wb") as encoder_file:
        pickle.dump(le, encoder_file)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(expected_metadata, metadata_file)
        




ENGLISH_TERMS = (
    "Terms and Conditions\n\n"
    "1. This application is for educational and informational purposes only.\n"
    "2. Estimates are based on machine learning and may not be accurate.\n"
    "3. Always consult a qualified doctor for medical advice and treatment.\n"
    "4. Do not make health decisions based on this application alone.\n"
    "5. Symptom selections and session notes are not saved or sent by the app. Predictions run locally.\n"
    "6. By clicking 'I Agree', you accept these terms."
)


class TermsPopup(tk.Toplevel):
    # What: modal popup that displays Terms & Conditions and requires user consent.
    # Why: enforce user acknowledgement before allowing access to the main functionality.
    # What it does: shows scrollable terms text and two buttons (I Agree / I Disagree).
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Terms and Conditions")
        self.geometry("600x600")
        self.resizable(False, False)
        self.grab_set()  # Make this window modal so user must interact with it

        # Scrollable text widget for terms
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        # Text widget shows the terms; scrollbar linked for long text.
        self.text = tk.Text(frame, wrap="word", yscrollcommand=scrollbar.set, font=("Segoe UI", 10))
        self.text.pack(fill="both", expand=True)
        scrollbar.config(command=self.text.yview)

        self.text.insert("1.0", ENGLISH_TERMS)
        self.text.config(state="disabled")  # Make text read-only so user cannot alter terms

        # Track whether the user agreed or not
        self.agreed = False

        # Button frame for both buttons (I Agree and I Disagree)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)

        # I Agree button: sets agreed flag and closes popup
        agree_btn = ttk.Button(btn_frame, text="I Agree", command=self.agree)
        agree_btn.pack(side="left", padx=10)

        # I Disagree button: optional behaviour (plays local video) then closes popup
        disagree_btn = ttk.Button(btn_frame, text="I Disagree", command=self.disagree)
        disagree_btn.pack(side="left", padx=10)

    def agree(self):
        # What: called when user clicks I Agree
        # Why: record consent and close popup so main app can open
        # What it does: sets flag and destroys this modal
        self.agreed = True
        self.destroy()

    def disagree(self):
        # What: called when user clicks I Disagree
        # Why: exit the app if user does not consent
        # What it does: so main block will exit the app.
        self.agreed = False
        self.destroy()


class SplashScreen(tk.Toplevel):
    def __init__(self, parent, logo_path='logo.png', duration=2000, size=300):
        super().__init__(parent)
        # Remove window decorations for a clean splash
        self.overrideredirect(True)
        self.configure(bg="#ffffff")
        self.duration = duration

        # Center the splash on screen
        w = size
        h = size
        parent.update_idletasks()
        x = (parent.winfo_screenwidth() // 2) - (w // 2)
        y = (parent.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

        # Canvas to hold logo and rotating arc
        self.canvas = tk.Canvas(self, width=w, height=h, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Try to load logo (PNG/GIF). Place in center.
        try:
            self.logo = tk.PhotoImage(file=logo_path)
            # scale down if bigger than available space
            lw = self.logo.width()
            if lw > (w - 40):
                factor = max(1, lw // (w - 40))
                self.logo = self.logo.subsample(factor, factor)
            self.canvas.create_image(w // 2, h // 2, image=self.logo)
        except Exception:
            # If logo not found or cannot be loaded, skip image
            pass

        # Arc that will rotate around the logo (white stroke)
        pad = 12
        self.arc = self.canvas.create_arc(pad, pad, w - pad, h - pad, start=0, extent=90,
                                          style='arc', outline="#0291be", width=6)
        # Use a darker background rectangle behind to make white arc visible
        self.canvas.create_rectangle(0, 0, w, h, outline='', fill='', tags='bg')
        self.angle = 0
        self._anim_id = None

    def _animate(self):
        # Rotate the arc by changing start angle
        self.angle = (self.angle + 8) % 360
        self.canvas.itemconfigure(self.arc, start=self.angle)
        self._anim_id = self.after(30, self._animate)

    def show(self):
        # Start animation and schedule close after duration; block until closed
        self._animate()
        self.after(self.duration, self.close)
        self.wait_window(self)

    def close(self):
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        self.destroy()


        
        
class FirstTimeUserDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Welcome to SymbiPredict")
        self.resizable(False, False)
        self.transient(parent)
        self.is_first_time = False

        ttk.Label(
            self,
            text="Is this your first time using SymbiPredict?",
            font=("Segoe UI", 15, "bold"),
            wraplength=380,
            justify="center",
        ).pack(padx=28, pady=(28, 10))
        ttk.Label(
            self,
            text="First-time users can take a quick guided tour before getting started.",
            wraplength=360,
            justify="center",
        ).pack(padx=28, pady=(0, 18))

        buttons = ttk.Frame(self)
        buttons.pack(pady=(0, 24))
        ttk.Button(
            buttons,
            text="Yes, show me",
            command=lambda: self.choose(True),
        ).pack(side="left", padx=6)
        ttk.Button(
            buttons,
            text="No, continue",
            command=lambda: self.choose(False),
        ).pack(side="left", padx=6)

        self.protocol("WM_DELETE_WINDOW", lambda: self.choose(False))
        self.grab_set()
        self.update_idletasks()
        x = (self.winfo_screenwidth() - self.winfo_width()) // 2
        y = (self.winfo_screenheight() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def choose(self, is_first_time):
        self.is_first_time = is_first_time
        self.destroy()


class TutorialDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("SymbiPredict quick guide")
        self.geometry("520x340")
        self.resizable(False, False)
        self.transient(parent)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Hint.TLabel", foreground="#637487", font=("Segoe UI", 9))
        style.configure("Section.TLabel", foreground="#173b57", font=("Segoe UI", 16, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        self.steps = (
            (
                "1. Find your symptoms",
                "Type a symptom in the search box, or browse the list. Names are shown in plain language.",
            ),
            (
                "2. Build your symptom list",
                "Select a symptom and choose Add symptom. Review your list on the right; you can remove one or clear the list.",
            ),
            (
                "3. View an estimate",
                "Choose Get estimate when your list is ready. This educational tool cannot diagnose illness or replace medical care.",
            ),
        )
        self.step_index = 0

        content = ttk.Frame(self, padding=28)
        content.pack(fill="both", expand=True)
        self.progress_label = ttk.Label(content, style="Hint.TLabel")
        self.progress_label.pack(anchor="w", pady=(0, 18))
        self.heading_label = ttk.Label(content, style="Section.TLabel", wraplength=440)
        self.heading_label.pack(anchor="w", pady=(0, 12))
        self.body_label = ttk.Label(content, wraplength=440, justify="left")
        self.body_label.pack(anchor="w", fill="x", expand=True)

        buttons = ttk.Frame(content)
        buttons.pack(fill="x", side="bottom", pady=(18, 0))
        self.back_button = ttk.Button(buttons, text="Back", command=self.previous_step)
        self.back_button.pack(side="left")
        self.next_button = ttk.Button(
            buttons,
            text="Next",
            style="Accent.TButton",
            command=self.next_step,
        )
        self.next_button.pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()
        self.update_idletasks()
        x = (self.winfo_screenwidth() - self.winfo_width()) // 2
        y = (self.winfo_screenheight() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        self.show_step()

    def show_step(self):
        heading, body = self.steps[self.step_index]
        self.progress_label.config(text=f"QUICK GUIDE  |  STEP {self.step_index + 1} OF {len(self.steps)}")
        self.heading_label.config(text=heading)
        self.body_label.config(text=body)
        self.back_button.config(state="normal" if self.step_index else "disabled")
        last_step = self.step_index == len(self.steps) - 1
        self.next_button.config(
            text="Start using SymbiPredict" if last_step else "Next",
            command=self.destroy if last_step else self.next_step,
        )

    def next_step(self):
        self.step_index += 1
        self.show_step()

    def previous_step(self):
        self.step_index -= 1
        self.show_step()


SYMPTOM_HELP = {
    "itching": "An uncomfortable urge to scratch the skin.",
    "skin_rash": "An area of skin that looks or feels irritated or changed.",
    "nodal_skin_eruptions": "Small raised bumps or eruptions on the skin.",
    "continuous_sneezing": "Repeated sneezing that continues for a period of time.",
    "chills": "Feeling unusually cold, sometimes with shivering.",
    "joint_pain": "Pain or discomfort in one or more joints.",
    "stomach_pain": "Pain or discomfort in the stomach area.",
    "acidity": "A burning or sour feeling in the upper stomach or chest.",
    "vomiting": "Bringing the contents of the stomach up through the mouth.",
    "fatigue": "Unusual tiredness or lack of energy.",
    "cough": "A reflex that clears or irritates the throat or airways.",
    "high_fever": "A measured body temperature that is higher than usual.",
    "mild_fever": "A slightly raised measured body temperature.",
    "breathlessness": "A feeling of difficulty or discomfort when breathing.",
    "dehydration": "A possible lack of enough fluid in the body; this label is not a self-diagnosis.",
    "headache": "Pain or discomfort in the head.",
    "nausea": "A queasy feeling, sometimes with an urge to vomit.",
    "loss_of_appetite": "Reduced desire to eat.",
    "diarrhoea": "Loose or watery bowel movements that occur more often than usual.",
    "constipation": "Difficulty passing stools or passing them less often than usual.",
    "yellowish_skin": "Skin that appears more yellow than usual.",
    "dark_urine": "Urine that appears darker than usual.",
    "runny_nose": "Fluid or mucus draining from the nose.",
    "congestion": "A blocked or stuffy feeling, often in the nose or chest.",
    "chest_pain": "Pain or discomfort in the chest; urgent symptoms need prompt medical attention.",
    "dizziness": "Feeling light-headed, unsteady, or as if the surroundings are moving.",
    "bruising": "A patch of discoloured skin after small blood vessels under the skin are injured.",
    "palpitations": "A noticeable feeling of a fast, pounding, or irregular heartbeat.",
    "loss_of_smell": "Reduced or absent ability to smell.",
    "redness_of_eyes": "Red or bloodshot appearance of the eyes.",
    "watering_from_eyes": "Eyes producing more tears than usual.",
    "swelling_joints": "Visible or felt swelling around one or more joints.",
    "muscle_pain": "Aches or soreness in one or more muscles.",
    "fever": "A measured body temperature that is higher than usual. This dataset does not distinguish high and mild fever.",
    "chills": "Feeling unusually cold, sometimes with shivering.",
    "nasal congestion": "A blocked or stuffy nose; people may describe this as a runny or congested nose.",
}

SYMPTOM_ALIASES = {
    "fever": ("high fever", "mild fever"),
    "chills": ("shivering",),
    "nasal congestion": ("runny nose", "stuffy nose"),
}


class GlossaryDialog(tk.Toplevel):
    def __init__(self, parent, symptoms, key_to_display):
        super().__init__(parent)
        self.title("Glossary")
        self.geometry("680x500")
        self.transient(parent)
        self.symptoms = symptoms
        self.key_to_display = key_to_display

        ttk.Label(self, text="Glossary", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", padx=18, pady=(16, 8)
        )
        self.search_var = tk.StringVar()
        search = ttk.Entry(self, textvariable=self.search_var)
        search.pack(fill="x", padx=18, pady=(0, 10))
        self.search_var.trace_add("write", self.update_list)

        area = ttk.Frame(self)
        area.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.listbox = tk.Listbox(area, width=35, font=("Segoe UI", 10), exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.show_definition)
        self.detail = ttk.Label(area, text="", wraplength=310, justify="left")
        self.detail.pack(side="left", fill="both", expand=True, padx=(16, 0), anchor="n")
        self.displayed_symptoms = []
        self.update_list()
        self.detail.config(text="Select a symptom to see a plain-language description. Wording can vary by region.")
        self.grab_set()

    def update_list(self, *args):
        query = self.search_var.get().strip().lower()
        self.listbox.delete(0, tk.END)
        self.displayed_symptoms = [
            symptom for symptom in self.symptoms if query in self.key_to_display[symptom].lower()
        ]
        for symptom in self.displayed_symptoms:
            self.listbox.insert(tk.END, self.key_to_display[symptom])

    def show_definition(self, event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        symptom = self.displayed_symptoms[selection[0]]
        heading = self.key_to_display[symptom]
        body = SYMPTOM_HELP.get(
            symptom,
            "This label comes from the model's dataset. If you are unsure what it means, do not guess; ask a qualified healthcare professional.",
        )
        self.detail.config(text=f"{heading}\n\n{body}")


class ReviewDialog(tk.Toplevel):
    def __init__(self, parent, symptoms, notes, duration, severity):
        super().__init__(parent)
        self.title("Review before estimate")
        self.geometry("540x500")
        self.resizable(False, False)
        self.transient(parent)
        self.confirmed = False
        ttk.Label(
            self,
            text="Review your choices before requesting an estimate.",
            font=("Segoe UI", 13, "bold"),
            wraplength=480,
        ).pack(anchor="w", padx=20, pady=(20, 10))
        symptoms_box = tk.Listbox(self, height=9, font=("Segoe UI", 10))
        symptoms_box.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        for symptom in symptoms:
            symptoms_box.insert(tk.END, symptom)
        notes_text = notes.strip() or "No personal notes"
        summary = (
            f"Duration: {duration}\n"
            f"Severity: {severity}\n\n"
            f"Personal notes (optional):\n{notes_text}"
        )
        ttk.Label(self, text=summary, wraplength=480, justify="left").pack(
            anchor="w", padx=20, pady=(0, 12)
        )
        ttk.Label(
            self,
            text="This is not a diagnosis. Please consult a qualified healthcare professional.",
            wraplength=480,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 12))
        emergency = "If symptoms are severe or rapidly worsening, including trouble breathing, chest pain, or loss of consciousness, contact local emergency services now."
        ttk.Label(self, text=emergency, wraplength=480, justify="left").pack(
            anchor="w", padx=20, pady=(0, 12)
        )
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=20, pady=(0, 20))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left")
        ttk.Button(
            buttons,
            text="Continue to estimate",
            command=self.confirm,
        ).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()

    def confirm(self):
        self.confirmed = True
        self.destroy()


class PrivacyDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Privacy")
        self.geometry("520x320")
        self.resizable(False, False)
        self.transient(parent)
        text = (
            "Session privacy\n\n"
            "• Symptom selections, severity, duration, and notes remain in memory while the app is open.\n"
            "• The app does not save or send these inputs; predictions run locally.\n"
            "• Notes, duration, and severity are not model inputs.\n"
            "• On first use, the app may save the model and label-encoder files locally."
        )
        ttk.Label(self, text=text, wraplength=470, justify="left").pack(
            fill="both", expand=True, padx=22, pady=22
        )
        ttk.Button(self, text="Close", command=self.destroy).pack(
            anchor="e", padx=22, pady=(0, 18)
        )
        self.grab_set()


class UnlistedSymptomDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Symptom not listed")
        self.geometry("440x220")
        self.resizable(False, False)
        self.transient(parent)
        self.value = None
        ttk.Label(
            self,
            text="Add an unlisted symptom to your private session notes. It will not be sent to the prediction model.",
            wraplength=390,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(20, 10))
        self.entry = ttk.Entry(self)
        self.entry.pack(fill="x", padx=20, pady=(0, 12))
        self.entry.bind("<Return>", lambda event: self.submit())
        ttk.Button(
            self,
            text="Add to notes",
            command=self.submit,
        ).pack(anchor="e", padx=20, pady=(0, 18))
        self.grab_set()
        self.entry.focus_set()

    def submit(self):
        self.value = self.entry.get().strip()
        if not self.value:
            messagebox.showinfo(
                "Symptom not listed",
                "Enter a short description of the symptom.",
                parent=self,
            )
            return
        self.destroy()


class StylishSymptomApp:
    # What: main Tkinter GUI application class
    # Why: provide a user interface to select symptoms and get disease predictions
    # What it does: constructs the layout, handles symptom selection, and runs predictions
    def __init__(self, root):
        self.root = root
        self.text_size = 10
        self.root.title("SymbiPredict - Symptom checker")
        self.root.geometry("1020x740")
        self.root.minsize(860, 640)
        self.root.configure(bg="#f3f6fb")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f6fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Header.TFrame", background="#173b57")
        style.configure("TLabel", background="#f3f6fb", foreground="#233447", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#173b57", foreground="#ffffff", font=("Segoe UI", 20, "bold"))
        style.configure("HeaderHint.TLabel", background="#173b57", foreground="#d8e8f2", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#ffffff", foreground="#173b57", font=("Segoe UI", 13, "bold"))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#637487", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(12, 9),
            background="#087e8b",
            foreground="#ffffff",
        )
        style.map("Accent.TButton", background=[("active", "#066873"), ("disabled", "#9aafb5")])
        style.configure("TEntry", font=("Segoe UI", 10), padding=7)

        self.key_to_display = {}
        self.display_to_key = {}
        for symptom in feature_columns:
            display_name = " ".join(symptom.replace("_", " ").split()).title()
            if display_name in self.display_to_key:
                suffix = 2
                while f"{display_name} ({suffix})" in self.display_to_key:
                    suffix += 1
                display_name = f"{display_name} ({suffix})"
            self.key_to_display[symptom] = display_name
            self.display_to_key[display_name] = symptom

        header = ttk.Frame(root, style="Header.TFrame", padding=(24, 18))
        header.pack(fill="x")
        header_copy = ttk.Frame(header, style="Header.TFrame")
        header_copy.pack(side="left", fill="x", expand=True)
        ttk.Label(header_copy, text="SymbiPredict", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header_copy,
            text="Explore symptom patterns with a simple, guided estimate.",
            style="HeaderHint.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        header_actions = ttk.Frame(header, style="Header.TFrame")
        header_actions.pack(side="right")
        ttk.Button(header_actions, text="Quick guide", command=self.show_tutorial).pack(side="left", padx=3)
        ttk.Button(header_actions, text="Glossary", command=self.show_glossary).pack(side="left", padx=3)
        ttk.Button(header_actions, text="Privacy", command=self.show_privacy).pack(side="left", padx=3)
        ttk.Button(
            header_actions,
            text="A−",
            command=lambda: self.change_text_size(-1),
        ).pack(side="left", padx=(8, 2))
        ttk.Button(
            header_actions,
            text="A+",
            command=lambda: self.change_text_size(1),
        ).pack(side="left", padx=2)

        # Main frame to hold left and right sections
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        ttk.Label(
            main_frame,
            text="Start by adding the symptoms you are experiencing.",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 14))

        # Left frame: Symptom selection and search
        left_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=16)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 9))

        ttk.Label(left_frame, text="Find a symptom", style="Section.TLabel").pack(anchor="w")
        ttk.Label(left_frame, text="Search by name, then select a result.", style="Hint.TLabel").pack(
            anchor="w", pady=(3, 10)
        )
        self.search_var = tk.StringVar()
        # Trace search_var to update symptom list dynamically as user types
        self.search_var.trace_add("write", self.update_symptom_list)
        search_entry = ttk.Entry(left_frame, textvariable=self.search_var)
        search_entry.pack(fill="x", pady=(0, 10))
        search_entry.bind("<Return>", lambda event: self.add_symptom())
        symptom_help_buttons = ttk.Frame(left_frame, style="Card.TFrame")
        symptom_help_buttons.pack(fill="x", pady=(0, 8))
        ttk.Button(
            symptom_help_buttons,
            text="Not sure",
            command=self.show_not_sure,
        ).pack(side="left")
        ttk.Button(
            symptom_help_buttons,
            text="My symptom isn't listed",
            command=self.add_unlisted_symptom,
        ).pack(side="right")

        available_area = ttk.Frame(left_frame, style="Card.TFrame")
        available_area.pack(fill="both", expand=True)
        available_scrollbar = ttk.Scrollbar(available_area)
        available_scrollbar.pack(side="right", fill="y")
        self.symptom_listbox = tk.Listbox(
            available_area,
            selectmode=tk.SINGLE,
            height=15,
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg="#233447",
            selectbackground="#087e8b",
            selectforeground="#ffffff",
            activestyle="none",
            highlightthickness=1,
            highlightbackground="#dce5ed",
            yscrollcommand=available_scrollbar.set,
        )
        self.symptom_listbox.pack(side="left", fill="both", expand=True)
        available_scrollbar.config(command=self.symptom_listbox.yview)
        self.symptom_listbox.bind("<Double-Button-1>", lambda event: self.add_symptom())
        self.update_symptom_list()  # Populate the symptom list initially

        add_btn = ttk.Button(left_frame, text="Add symptom", style="Accent.TButton", command=self.add_symptom)
        add_btn.pack(fill="x", pady=(12, 0))

        # Right frame: Selected symptoms and prediction controls
        right_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=16)
        right_frame.pack(side="right", fill="both", expand=True, padx=(9, 0))

        selected_heading = ttk.Frame(right_frame, style="Card.TFrame")
        selected_heading.pack(fill="x")
        ttk.Label(selected_heading, text="Your symptom list", style="Section.TLabel").pack(side="left")
        self.selection_count = tk.StringVar(value="0 added")
        ttk.Label(selected_heading, textvariable=self.selection_count, style="Hint.TLabel").pack(side="right")
        ttk.Label(
            right_frame,
            text="Review your choices before requesting an estimate.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 10))
        # Listbox showing symptoms user has selected for prediction
        selected_area = ttk.Frame(right_frame, style="Card.TFrame")
        selected_area.pack(fill="both", expand=True)
        selected_scrollbar = ttk.Scrollbar(selected_area)
        selected_scrollbar.pack(side="right", fill="y")
        self.selected_listbox = tk.Listbox(
            selected_area,
            selectmode=tk.SINGLE,
            height=15,
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg="#233447",
            selectbackground="#087e8b",
            selectforeground="#ffffff",
            activestyle="none",
            highlightthickness=1,
            highlightbackground="#dce5ed",
            yscrollcommand=selected_scrollbar.set,
        )
        self.selected_listbox.pack(side="left", fill="both", expand=True)
        selected_scrollbar.config(command=self.selected_listbox.yview)
        self.empty_selection_label = ttk.Label(
            selected_area,
            text="No symptoms added yet",
            style="Hint.TLabel",
        )
        self.empty_selection_label.place(relx=0.5, rely=0.5, anchor="center")

        # Remove single selected symptom button
        selection_buttons = ttk.Frame(right_frame, style="Card.TFrame")
        selection_buttons.pack(fill="x", pady=(10, 12))
        ttk.Button(selection_buttons, text="Remove selected", command=self.remove_symptom).pack(side="left")
        ttk.Button(selection_buttons, text="Clear list", command=self.remove_all_symptoms).pack(side="right")

        ttk.Label(right_frame, text="Personal notes (optional)", style="Section.TLabel").pack(
            anchor="w", pady=(2, 3)
        )
        ttk.Label(
            right_frame,
            text="Notes stay on this device for this session only; they are not saved or used for the estimate.",
            style="Hint.TLabel",
            wraplength=420,
        ).pack(anchor="w", pady=(0, 5))
        self.notes_var = tk.StringVar()
        notes_entry = ttk.Entry(right_frame, textvariable=self.notes_var)
        notes_entry.pack(fill="x", pady=(0, 6))
        notes_options = ttk.Frame(right_frame, style="Card.TFrame")
        notes_options.pack(fill="x", pady=(0, 8))
        self.duration_var = tk.StringVar(value="Not specified")
        self.duration_picker = ttk.Combobox(
            notes_options,
            textvariable=self.duration_var,
            values=("Not specified", "Less than a day", "1–3 days", "4–7 days", "More than a week", "Not sure"),
            state="readonly",
            width=18,
        )
        self.duration_picker.pack(side="left", padx=(0, 6))
        self.severity_var = tk.StringVar(value="Not specified")
        self.severity_picker = ttk.Combobox(
            notes_options,
            textvariable=self.severity_var,
            values=("Not specified", "Mild", "Moderate", "Severe", "Not sure"),
            state="readonly",
            width=14,
        )
        self.severity_picker.pack(side="left", padx=6)
        ttk.Button(notes_options, text="Clear notes", command=self.clear_notes).pack(side="right")

        # Predict button triggers model inference
        predict_btn = ttk.Button(
            right_frame,
            text="Get estimate",
            style="Accent.TButton",
            command=self.predict,
        )
        predict_btn.pack(fill="x")

        # Label to display prediction result and advisory message
        self.result_label = ttk.Label(
            right_frame,
            text="Your estimate will appear here.",
            wraplength=420,
            justify="left",
            padding=(0, 12, 0, 0),
        )
        self.result_label.pack(anchor="w", fill="x")
        ttk.Label(
            main_frame,
            text="For educational purposes only. This tool is not a medical diagnosis or a substitute for professional care.",
            style="Hint.TLabel",
            wraplength=960,
        ).pack(anchor="w", pady=(14, 0))

        self.root.bind_all("<Control-plus>", lambda event: self.change_text_size(1))
        self.root.bind_all("<Control-equal>", lambda event: self.change_text_size(1))
        self.root.bind_all("<Control-minus>", lambda event: self.change_text_size(-1))
        self.symptom_listbox.bind("<Return>", lambda event: self.add_symptom())
        self.selected_listbox.bind("<Delete>", lambda event: self.remove_symptom())
        self.refresh_selection()
        self.apply_text_size()
    def change_text_size(self, delta):
        self.text_size = max(9, min(18, self.text_size + delta))
        self.apply_text_size()

    def apply_text_size(self):
        style = ttk.Style(self.root)
        style.configure("TLabel", font=("Segoe UI", self.text_size))
        style.configure("Section.TLabel", font=("Segoe UI", self.text_size + 3, "bold"))
        style.configure("Hint.TLabel", font=("Segoe UI", max(9, self.text_size - 1)))
        style.configure("TButton", font=("Segoe UI", self.text_size))
        style.configure("Accent.TButton", font=("Segoe UI", self.text_size + 1, "bold"))
        self.symptom_listbox.configure(font=("Segoe UI", self.text_size))
        self.selected_listbox.configure(font=("Segoe UI", self.text_size))

    def show_glossary(self):
        dialog = GlossaryDialog(self.root, feature_columns, self.key_to_display)
        self.root.wait_window(dialog)

    def show_privacy(self):
        dialog = PrivacyDialog(self.root)
        self.root.wait_window(dialog)

    def show_not_sure(self):
        message = (
            "If you are unsure what a symptom means, do not guess. You can leave it out and ask a "
            "qualified healthcare professional.\n\n"
            "If symptoms are severe or rapidly worsening, including trouble breathing, chest pain, "
            "or loss of consciousness, contact local emergency services now."
        )
        messagebox.showinfo("Not sure", message, parent=self.root)

    def add_unlisted_symptom(self):
        dialog = UnlistedSymptomDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.value:
            current = self.notes_var.get().strip()
            self.notes_var.set(
                "; ".join(part for part in (current, f"Unlisted symptom: {dialog.value}") if part)
            )

    def clear_notes(self):
        self.notes_var.set("")
        self.duration_var.set("Not specified")
        self.severity_var.set("Not specified")

    def update_symptom_list(self, *args):
        # What: refreshes available symptoms shown in left listbox using current search string
        # Why: allows user to quickly find symptoms in a large list
        # What it does: clears listbox then inserts any feature_columns that match the search substring
        search = self.search_var.get().strip().lower()
        self.symptom_listbox.delete(0, tk.END)
        for symptom in feature_columns:
            display_name = self.key_to_display[symptom]
            aliases = SYMPTOM_ALIASES.get(symptom, ())
            if (
                search in symptom.lower()
                or search in display_name.lower()
                or any(search in alias for alias in aliases)
            ):
                self.symptom_listbox.insert(tk.END, display_name)

    def show_tutorial(self):
        tutorial = TutorialDialog(self.root)
        self.root.wait_window(tutorial)

    def refresh_selection(self):
        selected_count = self.selected_listbox.size()
        self.selection_count.set(f"{selected_count} added")
        if selected_count:
            self.empty_selection_label.place_forget()
        else:
            self.empty_selection_label.place(relx=0.5, rely=0.5, anchor="center")

    def add_symptom(self):
        # What: add the currently selected symptom from available list to selected listbox
        # Why: build a set of symptoms to send to the model for prediction
        # What it does: prevents duplicates in selected list
        selection = self.symptom_listbox.curselection()
        if not selection:
            return
        display_name = self.symptom_listbox.get(selection[0])
        if display_name not in self.selected_listbox.get(0, tk.END):
            self.selected_listbox.insert(tk.END, display_name)
            self.result_label.config(text="Your estimate will appear here.")
            self.refresh_selection()

    def remove_symptom(self):
        # What: remove one selected entry from the selected symptoms box
        # Why: allow user to correct selection without clearing everything
        # What it does: deletes the item at the selected index, if any
        selection = self.selected_listbox.curselection()
        if selection:
            self.selected_listbox.delete(selection[0])
            self.result_label.config(text="Your estimate will appear here.")
            self.refresh_selection()
    
    def remove_all_symptoms(self):
        # What: clear all selected symptoms
        # Why: provide a quick way to reset selections
        # What it does: deletes all items from the selected_listbox
        self.selected_listbox.delete(0, tk.END)
        self.result_label.config(text="Your estimate will appear here.")
        self.refresh_selection()

    def predict(self):
        # Predict the disease based on selected symptoms
        # What: creates a binary feature vector and calls the trained Keras model
        # Why: model expects fixed-length numeric input matching feature_columns order
        # What it does: displays predicted disease and model confidence in percent
        selected_names = self.selected_listbox.get(0, tk.END)
        if not selected_names:
            messagebox.showinfo(
                "Add symptoms first",
                "Please select at least one symptom before requesting an estimate.",
                parent=self.root,
            )
            return

        review = ReviewDialog(
            self.root,
            selected_names,
            self.notes_var.get(),
            self.duration_var.get(),
            self.severity_var.get(),
        )
        self.root.wait_window(review)
        if review.confirmed:
            self.run_prediction(selected_names)

    def run_prediction(self, selected_names):
        selected_symptoms = {self.display_to_key[name] for name in selected_names}
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in feature_columns]
        input_array = np.array([input_vector], dtype=np.float32)
        try:
            probabilities = model.predict(input_array, verbose=0)[0]
            top_classes = np.argsort(probabilities)[-3:][::-1]
            results = "\n".join(
                f"{le.classes_[class_index]} — {probabilities[class_index] * 100:.1f}% model score"
                for class_index in top_classes
            )
            self.result_label.config(
                text="Model-matched patterns (not diagnoses):\n"
                f"{results}\nThese scores are not clinically validated probabilities. "
                "Please consult a qualified healthcare professional."
            )
        except Exception as e:
            # On prediction error, show a messagebox so user knows what went wrong
            messagebox.showerror("Prediction Error", str(e), parent=self.root)




# Main entry point: control application startup and enforce Terms & Conditions consent
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide main window until terms are agreed

    # Show splash screen first (expects 'logo.png' next to this script; adjust path if needed)
    try:
        splash = SplashScreen(root, logo_path='logo.png', duration=2000, size=300)
        splash.show()
    except Exception:
        # If splash fails, continue to terms popup silently
        pass

    # What: show Terms popup modally and wait for it to close
    # Why: require explicit user consent before showing the main app
    popup = TermsPopup(root)
    root.wait_window(popup)  # Wait until the popup is closed

    # If user agreed, show the main application, otherwise exit
    if popup.agreed:
        app = StylishSymptomApp(root)
        root.deiconify()
        root.update_idletasks()

        first_time_dialog = FirstTimeUserDialog(root)
        root.wait_window(first_time_dialog)
        if first_time_dialog.is_first_time:
            tutorial = TutorialDialog(root)
            root.wait_window(tutorial)
        root.mainloop()
    else:
        root.destroy()  # Close the application if terms are not agreed
