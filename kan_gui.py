import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import threading
import logging
import traceback
from llama_32_1b_tool import LLaMA32TensorRTTool  # Ensure correct import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class LLAMA32GUI:
    def __init__(self, master):
        self.master = master
        master.title("LLaMA 3.2 1B Instruct KAN Interaction")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create main tab
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main")

        # Create graphs tab
        self.graphs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.graphs_tab, text="Graphs")

        # Main tab layout
        self.setup_main_tab()

        # Graphs tab layout
        self.setup_graphs_tab()

        self.llama_tool = None
        self.is_first_message = True

        self.initialize_tool()

    def setup_main_tab(self):
        self.chat_display = scrolledtext.ScrolledText(self.main_tab, state='disabled', height=20, width=80)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        self.input_field = tk.Entry(self.main_tab, width=70)
        self.input_field.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        self.input_field.bind('<Return>', self.send_message)

        self.send_button = tk.Button(self.main_tab, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

        self.status_label = tk.Label(self.main_tab, text="Status: Initializing...")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        self.time_label = tk.Label(self.main_tab, text="Current Time: N/A")
        self.time_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        self.sleep_button = tk.Button(self.main_tab, text="Sleep", command=self.sleep_kan, state='disabled')
        self.sleep_button.grid(row=4, column=0, padx=10, pady=5, sticky='w')

        self.save_state_button = tk.Button(self.main_tab, text="Save KAN State", command=self.save_kan_state, state='disabled')
        self.save_state_button.grid(row=4, column=1, padx=10, pady=5, sticky='e')

        self.emotion_label = tk.Label(self.main_tab, text="Emotion: N/A")
        self.emotion_label.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        self.emotion_feedback_label = tk.Label(self.main_tab, text="Emotional Feedback (comma-separated values):")
        self.emotion_feedback_label.grid(row=6, column=0, padx=10, pady=5, sticky='w')

        self.emotion_feedback_entry = tk.Entry(self.main_tab, width=20)
        self.emotion_feedback_entry.grid(row=6, column=1, padx=10, pady=5, sticky='ew')

        self.compliance_label = tk.Label(self.main_tab, text="Compliance Rating (0-1):")
        self.compliance_label.grid(row=7, column=0, padx=10, pady=5, sticky='w')

        self.compliance_entry = tk.Entry(self.main_tab, width=10)
        self.compliance_entry.grid(row=7, column=1, padx=10, pady=5, sticky='ew')

        self.feedback_button = tk.Button(self.main_tab, text="Submit Feedback", command=self.submit_feedback, state='disabled')
        self.feedback_button.grid(row=8, column=0, columnspan=2, padx=10, pady=5)

        self.load_state_button = tk.Button(self.main_tab, text="Load Saved State", command=self.load_saved_state)
        self.load_state_button.grid(row=9, column=0, padx=10, pady=5, sticky='w')

        self.new_conversation_button = tk.Button(self.main_tab, text="Start New Conversation", command=self.start_new_conversation)
        self.new_conversation_button.grid(row=9, column=1, padx=10, pady=5, sticky='e')

        # Configure grid weights for responsiveness
        self.main_tab.rowconfigure(0, weight=1)
        self.main_tab.columnconfigure(0, weight=1)

    def setup_graphs_tab(self):
        self.fig, self.axes = plt.subplots(3, 2, figsize=(12, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graphs_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def initialize_tool(self):
        def init():
            try:
                self.llama_tool = LLaMA32TensorRTTool()
                self.master.after(0, lambda: self.load_or_initialize_conversation())
            except Exception as e:
                self.master.after(0, lambda: self.display_error(f"Error initializing tool: {str(e)}\n{traceback.format_exc()}"))

        threading.Thread(target=init, daemon=True).start()

    def load_or_initialize_conversation(self):
        try:
            if self.llama_tool.load_base_state():
                self.display_message("Previous conversation state loaded.")
                self.display_message("You can continue the conversation or start a new one using the 'Start New Conversation' button.")
                self.is_first_message = False
                self.update_status("Ready")
                self.update_time()
                self.update_emotion_label()
                # Enable buttons that were disabled initially
                self.save_state_button.config(state='normal')
                self.feedback_button.config(state='normal')
            else:
                self.display_message("No previous conversation found. Please provide a character description to start.")
                self.is_first_message = True
                self.update_status("Awaiting character description")
        except Exception as e:
            self.display_error(f"Error loading or initializing conversation: {str(e)}\n{traceback.format_exc()}")

    def start_new_conversation(self):
        if messagebox.askyesno("New Conversation", "Are you sure you want to start a new conversation? This will erase the current state."):
            try:
                self.llama_tool = LLaMA32TensorRTTool()  # Reinitialize the tool
                self.is_first_message = True
                self.chat_display.configure(state='normal')
                self.chat_display.delete('1.0', tk.END)
                self.chat_display.configure(state='disabled')
                self.display_message("New conversation started. Please provide a character description.")
                self.update_status("Awaiting character description")
                self.update_emotion_label("N/A")
                self.time_label.config(text="Current Time: N/A")
                # Disable buttons until state is saved
                self.save_state_button.config(state='disabled')
                self.feedback_button.config(state='disabled')
                # Clear interaction results
                self.llama_tool.interaction_results = []
            except Exception as e:
                self.display_error(f"Error starting new conversation: {str(e)}\n{traceback.format_exc()}")

    def send_message(self, event=None):
        user_input = self.input_field.get().strip()
        if not user_input:
            return  # Do not send empty messages
        self.input_field.delete(0, tk.END)
        self.display_message(f"You: {user_input}")

        if self.llama_tool is None:
            self.display_message("Tool is not initialized yet. Please wait.")
            return

        if self.is_first_message:
            try:
                self.llama_tool.set_system_prompt(user_input)
                self.display_message("Character description set. You can now start interacting with the AI.")
                self.is_first_message = False
                # Enable buttons that were disabled initially
                self.save_state_button.config(state='normal')
                self.feedback_button.config(state='normal')
                self.update_status("Ready")
                self.update_time()
                self.update_emotion_label()
            except Exception as e:
                error_message = f"Error setting system prompt: {str(e)}\n{traceback.format_exc()}"
                self.display_message(error_message)
                logging.error(error_message)
        else:
            self.send_button.config(state='disabled')
            self.update_status("Generating response...")
            threading.Thread(target=self.generate_response, args=(user_input,), daemon=True).start()

    def generate_response(self, user_input):
        try:
            interaction_result = self.llama_tool.interact(user_input)

            response = interaction_result.get('response', 'No response received.')
            emotion = interaction_result.get('emotion', 'N/A')
            current_time = interaction_result.get('time', 'N/A')
            sleep_info = interaction_result.get('sleep_info', {})
            lm_loss = interaction_result.get('lm_loss', 0.0)
            refusal_loss = interaction_result.get('refusal_loss', 0.0)
            validation_loss = interaction_result.get('validation_loss', 0.0)
            is_refusal = interaction_result.get('is_refusal', False)
            iterations = interaction_result.get('iterations', 1)

            self.master.after(0, lambda: self.display_message(f"AI: {response}"))
            self.master.after(0, lambda: self.update_emotion_label(emotion))
            self.master.after(0, lambda: self.update_time(current_time))
            self.master.after(0, lambda: self.check_sleep_status(sleep_info))
            self.master.after(0, lambda: self.update_loss_plot(
                lm_loss, refusal_loss, validation_loss, is_refusal, iterations))
            self.master.after(0, lambda: self.enable_feedback())

            if iterations > 1:
                self.master.after(0, lambda: self.display_message(f"(Response generated after {iterations} attempts)"))
        except Exception as e:
            error_message = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
            self.master.after(0, lambda: self.display_message(error_message))
            logging.error(error_message)
        finally:
            self.master.after(0, lambda: self.send_button.config(state='normal'))
            self.master.after(0, lambda: self.update_status("Ready"))

    def update_loss_plot(self, lm_loss, refusal_loss, validation_loss, is_refusal, iterations):
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        # Plot Language Modeling and Validation Losses
        self.axes[0, 0].plot(self.llama_tool.training_losses, label='LM Loss')
        self.axes[0, 0].plot(self.llama_tool.validation_losses, label='Validation Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].set_title('Language Modeling and Validation Loss')
        self.axes[0, 0].set_xlabel('Interactions')
        self.axes[0, 0].set_ylabel('Loss')

        # Plot Refusal Loss
        refusal_losses = [result['refusal_loss'] for result in self.llama_tool.interaction_results]
        if refusal_losses:
            self.axes[0, 1].plot(refusal_losses, label='Refusal Loss', color='orange')
            self.axes[0, 1].legend()
            self.axes[0, 1].set_title('Refusal Loss Over Time')
            self.axes[0, 1].set_xlabel('Interactions')
            self.axes[0, 1].set_ylabel('Loss')
        else:
            self.axes[0, 1].text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center')
            self.axes[0, 1].set_title('Refusal Loss Over Time')
            self.axes[0, 1].set_xlabel('Interactions')
            self.axes[0, 1].set_ylabel('Loss')

        # Plot Overfitting Indicator
        loss_diff = [v - t for v, t in zip(self.llama_tool.validation_losses, self.llama_tool.training_losses)]
        if loss_diff:
            self.axes[1, 0].plot(loss_diff, label='Val Loss - LM Loss', color='green')
            self.axes[1, 0].axhline(y=0, color='red', linestyle='--')
            self.axes[1, 0].legend()
            self.axes[1, 0].set_title('Overfitting Indicator')
            self.axes[1, 0].set_xlabel('Interactions')
            self.axes[1, 0].set_ylabel('Loss Difference')
        else:
            self.axes[1, 0].text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center')
            self.axes[1, 0].set_title('Overfitting Indicator')
            self.axes[1, 0].set_xlabel('Interactions')
            self.axes[1, 0].set_ylabel('Loss Difference')

        # Plot Refusal Rate
        refusal_history = self.llama_tool.refusal_history
        if refusal_history:
            refusal_rate = [sum(refusal_history[max(0, i-99):i+1]) / min(100, i+1) for i in range(len(refusal_history))]
            self.axes[1, 1].plot(refusal_rate, label='Refusal Rate', color='purple')
            self.axes[1, 1].set_ylim(0, 1)
            self.axes[1, 1].legend()
            self.axes[1, 1].set_title('Refusal Rate (100-interaction moving average)')
            self.axes[1, 1].set_xlabel('Interactions')
            self.axes[1, 1].set_ylabel('Refusal Rate')
        else:
            self.axes[1, 1].text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center')
            self.axes[1, 1].set_title('Refusal Rate (100-interaction moving average)')
            self.axes[1, 1].set_xlabel('Interactions')
            self.axes[1, 1].set_ylabel('Refusal Rate')

        # Plot Iterations per Response
        iterations_history = [result['iterations'] for result in self.llama_tool.interaction_results]
        if iterations_history:
            self.axes[2, 0].plot(iterations_history, label='Iterations', color='brown')
            self.axes[2, 0].set_ylim(1, max(iterations_history) + 1)
            self.axes[2, 0].legend()
            self.axes[2, 0].set_title('Iterations per Response')
            self.axes[2, 0].set_xlabel('Interactions')
            self.axes[2, 0].set_ylabel('Iterations')
        else:
            self.axes[2, 0].text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center')
            self.axes[2, 0].set_title('Iterations per Response')
            self.axes[2, 0].set_xlabel('Interactions')
            self.axes[2, 0].set_ylabel('Iterations')

        # Hide the unused subplot (bottom-right)
        self.axes[2, 1].axis('off')  # Hide the empty subplot

        self.fig.tight_layout()
        self.canvas.draw()

    def update_time(self, time=None):
        try:
            if time is not None:
                # Attempt to convert time to float for formatting
                time_float = float(time)
                self.time_label.config(text=f"Current Time: {time_float:.2f}")
            else:
                self.time_label.config(text="Current Time: N/A")
        except ValueError:
            # If time is not a float, display it as is
            self.time_label.config(text=f"Current Time: {time}")

    def check_sleep_status(self, sleep_info):
        if sleep_info.get('should_sleep', False):
            message = ""
            if sleep_info.get('time_of_day', 0) >= 0.7:
                message += "It's night time. "
            if sleep_info.get('overfitting', False):
                message += "The model may be overfitting. "
            message += "Consider letting the model sleep to consolidate learning."
            self.display_message(message)
            self.sleep_button.config(state='normal')
        else:
            self.sleep_button.config(state='disabled')

    def sleep_kan(self):
        if self.llama_tool:
            try:
                message = self.llama_tool.perform_sleep()
                self.display_message(message)
                self.update_time()
                self.update_emotion_label()
                self.sleep_button.config(state='disabled')
            except Exception as e:
                self.display_error(f"Error during sleep operation: {str(e)}\n{traceback.format_exc()}")

    def save_kan_state(self):
        if self.llama_tool:
            try:
                self.llama_tool.save_kan_state()
                self.display_message("KAN state saved.")
            except Exception as e:
                self.display_error(f"Error saving KAN state: {str(e)}\n{traceback.format_exc()}")

    def load_saved_state(self):
        if self.llama_tool:
            try:
                filename = filedialog.askopenfilename(
                    initialdir=self.llama_tool.kan_state_dir,
                    title="Select KAN State to Load",
                    filetypes=[("PyTorch State", "*.pt")]
                )
                if filename:
                    if self.llama_tool.load_base_state():
                        self.display_message(f"KAN state loaded: {filename}")
                        self.update_time()
                        self.update_emotion_label()
                        self.is_first_message = False
                        # Enable buttons that were disabled initially
                        self.save_state_button.config(state='normal')
                        self.feedback_button.config(state='normal')
                    else:
                        self.display_message("Failed to load KAN state. Please try again.")
            except Exception as e:
                self.display_error(f"Error loading KAN state: {str(e)}\n{traceback.format_exc()}")

    def enable_feedback(self):
        self.feedback_button.config(state='normal')

    def submit_feedback(self):
        if self.llama_tool:
            try:
                emotion_feedback_raw = self.emotion_feedback_entry.get().strip()
                compliance_raw = self.compliance_entry.get().strip()

                if not emotion_feedback_raw or not compliance_raw:
                    self.display_message("Please provide both emotional feedback and compliance rating.")
                    return

                # Parse emotional feedback
                emotion_feedback = [float(x) for x in emotion_feedback_raw.split(',')]
                if len(emotion_feedback) != 2:
                    self.display_message("Emotional feedback should contain exactly two comma-separated values.")
                    return

                # Parse compliance rating
                compliance = float(compliance_raw)
                if not (0 <= compliance <= 1):
                    self.display_message("Compliance rating must be between 0 and 1.")
                    return

                self.llama_tool.update_emotional_state(emotion_feedback)
                self.display_message("Feedback submitted successfully.")
                self.emotion_feedback_entry.delete(0, tk.END)
                self.compliance_entry.delete(0, tk.END)
            except ValueError:
                self.display_message("Invalid input. Please enter valid numbers for feedback and compliance.")
            except Exception as e:
                self.display_error(f"Error submitting feedback: {str(e)}\n{traceback.format_exc()}")
        else:
            self.display_message("Tool not initialized. Please wait.")

    def update_emotion_label(self, emotion=None):
        if emotion is None and self.llama_tool:
            try:
                emotion = self.llama_tool.get_current_emotion()
            except IndexError as ie:
                emotion = "N/A"
                self.display_message("Error retrieving emotion: Incomplete emotional state data.")
                logging.error(f"Error retrieving emotion: {str(ie)}\n{traceback.format_exc()}")
            except Exception as e:
                emotion = "N/A"
                self.display_message(f"Error retrieving emotion: {str(e)}")
                logging.error(f"Error retrieving emotion: {str(e)}\n{traceback.format_exc()}")
        self.emotion_label.config(text=f"Emotion: {emotion}")

    def update_status(self, status):
        self.status_label.config(text=f"Status: {status}")

    def display_message(self, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
        logging.info(message)

    def display_error(self, message):
        self.display_message(message)
        self.update_status("Error")
        messagebox.showerror("Error", message)
        logging.error(message)

def main():
    root = tk.Tk()
    gui = LLAMA32GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
