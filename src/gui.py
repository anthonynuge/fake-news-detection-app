import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog


class App(tk.Tk):
    def __init__(self, model, sentiment_analyzer):
        # main setup
        super().__init__()
        self.title("Real or Fake News App")
        self.geometry("800x400")
        self.minsize(400, 200)

        # widgets frames
        self.target = Entry(self)
        self.analysis = Analysis(self)
        self.menu = Menu(self)

        # Loaded model
        self.model = model
        self.sia = sentiment_analyzer

        # Run
        self.mainloop()

    def classify_news(self):
        news_input = self.target.get_text()
        # Classify news article if there is text.
        if news_input:
            classification = self.model.predict([news_input])[0]
            classification = "Fake" if classification == 1 else "Real"
            print(f"Result: {classification}")
            sentiment = self.sia.get_sentiment(news_input)
            print(f"Sentiment: {sentiment}")
            print("=" * 30)
            self.analysis.update_analysis(sentiment, classification)
            # self.analysis.update_news_type(classification)
            # self.analysis.update_sentiment(sentiment)

    # Browse text file and load content into the text area
    def browse_files(self):
        file = filedialog.askopenfile(
            initialdir="~",
            title="Select a file",
            filetypes=(("Text files", "*.txt*"), ("all files", "*.*")),
        )
        if file is not None:
            print("Selected File: " + file.name)
            content = file.read()
            self.target.add_file_contents(content)
        else:
            print("No file selected")

    def clear_all(self):
        self.target.clear_text()
        self.analysis.clear_analysis()


class Menu(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.entry_instance = parent.target
        self.analysis_instance = parent.analysis
        self.app = parent
        self.create_widgets()
        self.place(relx=0.8, y=0, relheight=0.80, relwidth=0.20)

    def create_widgets(self):
        browse_btn = ttk.Button(
            self, text="Browse Files", command=self.app.browse_files
        )
        run_btn = ttk.Button(self, text="Predict", command=self.app.classify_news)
        clear_btn = ttk.Button(self, text="Clear", command=self.app.clear_all)
        browse_btn.pack()
        run_btn.pack()
        clear_btn.pack()


class Entry(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.place(x=0, y=0, relwidth=0.8, relheight=0.8)

        label = ttk.Label(self, text="Check if news article is real or fake:")
        self.text_area = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1
        )

        label.pack(expand=True, fill="both")
        self.text_area.pack(expand=True, fill="both")

    def add_file_contents(self, file_contents):
        self.clear_text()
        self.text_area.insert(tk.END, file_contents)

    def get_text(self):
        return self.text_area.get(1.0, "end-1c")

    def clear_text(self):
        self.text_area.delete(1.0, tk.END)


class Analysis(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.place(x=0, rely=0.80, relwidth=1.0, relheight=0.20)

        # grid layout
        self.columnconfigure((0, 1), weight=1, uniform="a")
        self.rowconfigure((0), weight=1)

        # Output Labels:
        self.sentiment_label = ttk.Label(self, text="Sentiment: ")
        self.news_type_label = ttk.Label(self, text="Prediciton: ")

        self.sentiment_label.grid(row=0, column=1)
        self.news_type_label.grid(row=0, column=0)

    def clear_analysis(self):
        self.sentiment_label["text"] = "Sentiment: "
        self.news_type_label["text"] = "Prediction: "

    def update_analysis(self, sentiment, prediction):
        self.sentiment_label["text"] = f"Sentiment: {sentiment}"
        self.news_type_label["text"] = f"Prediction: {prediction}"

    def update_news_type(self, classification):
        self.news_type_label["text"] = f"Prediction: {classification}"

    def update_sentiment(self, sentiment):
        self.sentiment_label["text"] = f"Prediction: {sentiment}"


if __name__ == "__main__":
    pass
