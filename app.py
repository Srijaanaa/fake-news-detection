import os
import pickle
import pandas as pd
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, desc, asc
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sqlalchemy import and_, or_ 

# --- Matplotlib Fix for Threading/GUI Issues ---
# Set the backend to 'Agg' (non-GUI) BEFORE importing any plotting submodules.
import matplotlib
matplotlib.use('Agg')
# -----------------------------------------------

# Import custom modules
from database import User, NewsHistory, Base, engine, Session
from utils import preprocess_text, quick_sort, binary_search
from report import generate_metrics, generate_confusion_matrix_plot, generate_metrics_bar_chart

# Initialize Flask application
app = Flask(__name__)

# Use environment variable for secret key; fallback only for local dev
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')

# Use UPLOAD_FOLDER env var or place uploads inside the app root
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join(app.root_path, 'uploads'))
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure required directories exist inside app.root_path
os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'model'), exist_ok=True)

# Use absolute paths based on app.root_path for safety
TRAINING_PROGRESS_FILE = os.path.join(app.root_path, 'model', 'training_progress.json')
MODEL_PATH = os.path.join(app.root_path, 'model', 'logistic_regression_model.pkl')
VECTORIZER_PATH = os.path.join(app.root_path, 'model', 'tfidf_vectorizer.pkl')
METRICS_PATH = os.path.join(app.root_path, 'model', 'metrics.json')

# Create a database session: MUST use scoped_session for Flask thread safety
db_session = scoped_session(Session)

# Load the pre-trained model and vectorizer
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Warning: Model files not found. Please run 'model_training.py' or retrain the model.")
    model = None
    tfidf_vectorizer = None

@app.before_request
def before_request():
    """Ensure database connection is open before each request."""
    pass

@app.teardown_request
def teardown_request(exception=None):
    """Close the database session after each request."""
    db_session.remove()

def is_authenticated():
    """Helper function to check if a user is logged in."""
    return 'user_id' in session

def get_current_user_id():
    """Helper function to get the current user's ID."""
    return session.get('user_id')

def is_admin():
    """Helper function to check if the current user has the 'admin' role."""
    user_id = get_current_user_id()
    if user_id:
        user = db_session.query(User).filter_by(id=user_id).first()
        # Fallback to user_id == 1 if the user object couldn't be retrieved for some reason
        if user:
            return user.role == 'admin'
        # Fallback for old sessions/initial admin if no user object is retrieved
        return user_id == 1
    return False
@app.context_processor
def inject_user_helpers():
    return dict(
        is_authenticated=is_authenticated,
        is_admin=is_admin
    )
def load_metrics_from_file():
    """Loads model metrics from the saved JSON file using an absolute path."""
    
    # Use os.path.join and app.root_path to construct a reliable absolute path
    absolute_metrics_path = os.path.join(app.root_path, METRICS_PATH)
    
    try:
        with open(absolute_metrics_path, 'r') as f:
            # Check if the file is empty before loading
            content = f.read()
            if not content:
                print("Warning: metrics.json is empty.")
                return {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
                }
            
            # Reset file pointer and load content
            f.seek(0) 
            return json.loads(content)
            
    except FileNotFoundError:
        print(f"Metrics file not found at: {absolute_metrics_path}")
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
        }
    except json.JSONDecodeError as e:
        print(f"Error decoding metrics.json: {e}")
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
        }
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
        }
   

def update_training_status(progress, message, is_running):
    """Updates the training status file for the client to poll."""
    status = {'progress': progress, 'message': message, 'is_running': is_running}
    try:
        with open(TRAINING_PROGRESS_FILE, 'w') as f:
            json.dump(status, f)
    except Exception as e:
        print(f"Error writing training status file: {e}")

def get_training_status():
    """Reads the training status file."""
    try:
        with open(TRAINING_PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'progress': 100, 'message': 'Ready', 'is_running': False}
    except Exception as e:
        print(f"Error reading training status file: {e}")
        return {'progress': 0, 'message': 'Error reading status', 'is_running': False}


def retrain_model():
    """
    Loads data, trains the model, saves the model/vectorizer, and updates training status.
    """
    global model, tfidf_vectorizer
    
    # Check if a training process is already running via the status file
    if get_training_status().get('is_running'):
        print("Training already running. Aborting new request.")
        return False

    update_training_status(5, 'Starting training...', True)

    try:
        # --- 1. Load Data (5% -> 25%) ---
        fake_df = pd.read_csv('data/Fake.csv')
        true_df = pd.read_csv('data/True.csv')
        
        # Assign labels before concatenation
        fake_df['label'] = 0
        true_df['label'] = 1
        
        df = pd.concat([fake_df, true_df]).reset_index(drop=True)
        df = df.dropna(subset=['text', 'label'])
        
        X = df['text']
        y = df['label']
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        update_training_status(25, 'Data loaded and split. Starting TF-IDF Vectorization...', True)

        # --- 2. TF-IDF Vectorization (25% -> 50%) ---
        
        # Reinitialize vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.apply(preprocess_text))
        X_test_tfidf = tfidf_vectorizer.transform(X_test.apply(preprocess_text))

        update_training_status(50, 'Vectorization complete. Training Logistic Regression model...', True)

        # --- 3. Model Training (50% -> 75%) ---
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        # --- 4. Evaluation and Saving (75% -> 100%) ---
        update_training_status(75, 'Model trained. Generating metrics and charts...', True)
        
        y_pred = model.predict(X_test_tfidf)
        metrics = generate_metrics(y_test, y_pred)
        
        # Save model and vectorizer
        with open(MODEL_PATH, 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(VECTORIZER_PATH, 'wb') as vectorizer_file:
            pickle.dump(tfidf_vectorizer, vectorizer_file)

        # Save metrics to JSON file
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f)
            
        # Generate and save charts (NOTE: Report functions must save to static/images/)
        # This assumes your generate_... functions are correctly configured to save 
        # to the 'static/images' folder, e.g., 'static/images/confusion_matrix.png'
        generate_confusion_matrix_plot(y_test, y_pred, output_path='static/images/confusion_matrix.png')
        generate_metrics_bar_chart(metrics, output_path='static/images/metrics_bar_chart.png')
        
        update_training_status(100, 'Training complete! Reloading Dashboard.', False)
        return True

    except FileNotFoundError:
        update_training_status(0, 'Training failed: Data files not found.', False)
        print("Training failed: data files not found.")
        return False
    except Exception as e:
        update_training_status(0, f'Training failed: {e}', False)
        print(f"An error occurred during training: {e}")
        return False

# --- CORE APP ROUTES (Index, Login, Detect, History, Dashboard, Profile) ---

@app.route('/')
def index():
    """Render the main fake news detection page."""
    if not is_authenticated():
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user signup."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password are required!', 'danger')
            return redirect(url_for('signup'))
        
        try:
            # Determine if this is the first user (for admin assignment)
            is_first_user = db_session.query(User).count() == 0
            initial_role = 'admin' if is_first_user else 'user'
            
            password_hash = generate_password_hash(password)
            
            new_user = User(
                username=username, 
                password_hash=password_hash, 
                role=initial_role
            ) 
            
            db_session.add(new_user)
            db_session.commit()
            
            # Auto-login the new user
            session['user_id'] = new_user.id
            
            if is_first_user:
                 flash('Welcome Admin! Your account was automatically set to Administrator role.', 'success')
            else:
                 flash('Account created successfully! You are now logged in.', 'success')
            
            return redirect(url_for('index'))
            
        except IntegrityError:
            db_session.rollback()
            flash('Username already exists. Please choose another.', 'danger')
            
        except Exception as e:
            # CRITICAL: This catch-all block is essential for troubleshooting silent commit failures.
            db_session.rollback()
            print(f"*** CRITICAL DATABASE ERROR during signup: {e} ***") 
            flash(f"A system error occurred during signup. Check console for details.", 'danger') 
            
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = db_session.query(User).filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handle user logout."""
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    """Handles user profile viewing and updates."""
    if not is_authenticated():
        return redirect(url_for('login'))

    user_id = get_current_user_id()
    user = db_session.query(User).filter_by(id=user_id).first()

    if not user:
        flash("User profile not found. Please log in again.", "danger")
        return redirect(url_for('logout'))

    if request.method == 'POST':
        new_username = request.form.get('username')
        new_password = request.form.get('new_password')
        current_password = request.form.get('current_password')

        if not check_password_hash(user.password_hash, current_password):
            flash("Incorrect current password.", "danger")
            return redirect(url_for('profile'))

        changes_made = False

        # 1. Update Username
        if new_username and new_username != user.username:
            try:
                user.username = new_username
                db_session.commit()
                flash("Username updated successfully.", "success")
                changes_made = True
            except IntegrityError:
                db_session.rollback()
                flash("That username is already taken.", "danger")
            except Exception as e:
                db_session.rollback()
                flash("Error updating username.", "danger")
                print(f"Error updating username: {e}")

        # 2. Update Password
        if new_password:
            user.password_hash = generate_password_hash(new_password)
            db_session.commit()
            flash("Password updated successfully.", "success")
            changes_made = True
        
        if not changes_made:
            flash("No changes were made.", "info")

        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)


@app.route('/detect', methods=['POST'])
def detect_news():
    """Handles news detection."""
    if not is_authenticated():
        return redirect(url_for('login'))
        
    news_text = request.form.get('news_text')
    
    if not model or not tfidf_vectorizer:
        flash("Model is not loaded. Please train the model first.", "danger")
        return redirect(url_for('index'))
        
    if news_text:
        processed_text = preprocess_text(news_text)
        text_vectorized = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized).max()
        prediction_label = 'Real' if prediction == 1 else 'Fake'
        
        new_detection = NewsHistory(
            user_id=get_current_user_id(),
            news_text=news_text,
            prediction=prediction_label,
            confidence=confidence
        )
        db_session.add(new_detection)
        db_session.commit()
        
        return render_template('index.html', result=prediction_label, confidence=f"{confidence*100:.2f}%")
    
    return redirect(url_for('index'))

@app.route('/bulk_upload', methods=['POST'])
def bulk_upload():
    """Handles bulk CSV file upload and news classification."""
    if not is_authenticated():
        return redirect(url_for('login'))
        
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
        
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Error reading CSV file: {e}", "danger")
            return redirect(url_for('index'))
            
        text_column = 'text' if 'text' in df.columns else 'title'
        if text_column not in df.columns:
            flash("CSV must contain a 'text' or 'title' column.", "danger")
            return redirect(url_for('index'))
            
        if not model or not tfidf_vectorizer:
            flash("Model is not loaded. Please train the model first.", "danger")
            return redirect(url_for('index'))
            
        results = []
        for index, row in df.iterrows():
            news_text = str(row[text_column])
            processed_text = preprocess_text(news_text)
            
            text_vectorized = tfidf_vectorizer.transform([processed_text])
            prediction = model.predict(text_vectorized)[0]
            confidence = model.predict_proba(text_vectorized).max()
            prediction_label = 'Real' if prediction == 1 else 'Fake'
            
            new_detection = NewsHistory(
                user_id=get_current_user_id(),
                news_text=news_text,
                prediction=prediction_label,
                confidence=confidence
            )
            db_session.add(new_detection)
            
            results.append({
                'text': news_text,
                'prediction': prediction_label,
                'confidence': f"{confidence*100:.2f}%"
            })
            
        db_session.commit()
        flash("Bulk upload and classification completed!", "success")
        return render_template('results.html', results=results)
        
    flash('Invalid file type. Please upload a CSV file.', 'danger')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Generates and displays model evaluation metrics and charts."""
    if not is_authenticated():
        return redirect(url_for('login'))
        
    metrics = load_metrics_from_file()

    # Check for file existence in the correct 'static/images' subfolder
    chart_exists = (
        os.path.exists('static/images/metrics_bar_chart.png') and 
        os.path.exists('static/images/confusion_matrix.png')
    )
    
    # Check if metrics are zero (if the file was found but contained zeros)
    metrics_are_zero = metrics.get('accuracy', 0.0) == 0.0
    
    if not chart_exists or metrics_are_zero:
        flash("Charts and metrics files not found or contain no data. Please train the model.", "warning")
        
    return render_template('dashboard.html', metrics=metrics, has_charts=chart_exists)

@app.route('/history', methods=['GET', 'POST'])
def history():
    """Retrieves and displays news history with sorting and searching."""
    if not is_authenticated():
        return redirect(url_for('login'))
        
    user_id = get_current_user_id()
    detections = db_session.query(NewsHistory).filter_by(user_id=user_id).order_by(desc(NewsHistory.timestamp)).all()
    
    sort_by = request.args.get('sort', 'date_desc')
    search_id = request.args.get('search_id')
    
    if search_id:
        try:
            search_id = int(search_id)
            sorted_detections = sorted(detections, key=lambda x: x.id)
            found_index = binary_search([d.id for d in sorted_detections], search_id)
            
            if found_index != -1:
                found_item = sorted_detections[found_index]
                flash(f"Found news detection with ID {found_item.id}.", "success")
                detections = [found_item]
            else:
                flash(f"No news detection found with ID {search_id}.", "warning")
                detections = []
        except ValueError:
            flash("Invalid search ID.", "danger")
    
    if sort_by == 'date_asc':
        detections = sorted(detections, key=lambda x: x.timestamp)
    elif sort_by == 'date_desc':
        detections = sorted(detections, key=lambda x: x.timestamp, reverse=True)
    elif sort_by == 'confidence_asc':
        detections = sorted(detections, key=lambda x: x.confidence)
    elif sort_by == 'confidence_desc':
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    return render_template('history.html', detections=detections)

@app.route('/history/delete/<int:item_id>', methods=['POST'])
def delete_history(item_id):
    """Deletes a specific history item."""
    if not is_authenticated():
        return redirect(url_for('login'))
        
    user_id = get_current_user_id()
    item_to_delete = db_session.query(NewsHistory).filter_by(id=item_id, user_id=user_id).first()
    
    if item_to_delete:
        db_session.delete(item_to_delete)
        db_session.commit()
        flash("Detection history deleted successfully.", "success")
    else:
        flash("Item not found or you don't have permission to delete it.", "danger")
        
    return redirect(url_for('history'))

# --- ADMIN ROUTES ---

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard to view user stats and manage training."""
    if not is_authenticated() or not is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    # Get all users and their current roles for display
    all_users = db_session.query(User).all()
    total_users = len(all_users)
    
    # Fetch user metrics
    all_history = db_session.query(NewsHistory).all()
    user_data = {}

    for history in all_history:
        user_id = history.user_id
        if user_id not in user_data:
            user_data[user_id] = {'total_confidence': 0, 'count': 0}
        user_data[user_id]['total_confidence'] += history.confidence
        user_data[user_id]['count'] += 1

    user_stats = {}
    for user in all_users:
        stats = user_data.get(user.id, {'total_confidence': 0, 'count': 0})
        
        avg_confidence = 0
        if stats['count'] > 0:
            avg_confidence = (stats['total_confidence'] / stats['count'])
        
        user_stats[user.id] = {
            'username': user.username,
            'role': user.role, 
            'total_predictions': stats['count'],
            'avg_confidence': f"{avg_confidence * 100:.2f}%"
        }
    
    # Pass current training status to the template
    return render_template('admin_dashboard.html', 
                           total_users=total_users, 
                           user_stats=user_stats,
                           initial_status=get_training_status())

@app.route('/admin/retrain', methods=['POST'])
def admin_retrain():
    """Admin route to trigger model retraining. This runs synchronously."""
    if not is_authenticated() or not is_admin():
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
    
    # Start the training process (which updates the status file)
    success = retrain_model() 

    if success:
        # Client side polling will handle the success message and reload
        return jsonify(get_training_status()), 200
    else:
        status = get_training_status()
        return jsonify(status), 500

@app.route('/api/training_progress')
def api_training_progress():
    """API endpoint to poll for model training status."""
    # Authenticate the request, but return a simple JSON error if failed
    if not is_authenticated() or not is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    
    return jsonify(get_training_status())

@app.route('/admin/update_role/<int:user_id>', methods=['POST'])
def admin_update_role(user_id):
    """Admin route to change a user's role."""
    if not is_authenticated() or not is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    new_role = request.form.get('new_role')
    
    if new_role not in ['user', 'admin']:
        flash('Invalid role specified.', 'danger')
        return redirect(url_for('admin_dashboard'))

    user = db_session.query(User).filter_by(id=user_id).first()
    
    if not user:
        flash(f'Error: User ID {user_id} not found.', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    user.role = new_role
    db_session.commit()
    
    flash(f'Successfully updated role for {user.username} to {new_role}.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_history/<int:user_id>', methods=['POST'])
def admin_delete_history(user_id):
    """Admin route to delete all prediction history for a specific user."""
    if not is_authenticated() or not is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    deleted_count = db_session.query(NewsHistory).filter_by(user_id=user_id).delete()
    db_session.commit()

    user = db_session.query(User).filter_by(id=user_id).first()
    username = user.username if user else f"User ID {user_id}"
    
    flash(f'Successfully deleted {deleted_count} history records for user: {username}.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    """Admin route to delete a user and all their associated data."""
    if not is_authenticated() or not is_admin():
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    # Prevent admin from deleting themselves
    if user_id == get_current_user_id():
        flash('Error: You cannot delete your own active account.', 'danger')
        return redirect(url_for('admin_dashboard'))

    user = db_session.query(User).filter_by(id=user_id).first()
    
    if user:
        username = user.username
        
        # 1. Delete all associated NewsHistory
        db_session.query(NewsHistory).filter_by(user_id=user_id).delete()
        
        # 2. Delete the User record
        db_session.delete(user)
        db_session.commit()
        
        flash(f'Successfully deleted user: {username} and all their data.', 'success')
    else:
        flash(f'Error: User ID {user_id} not found.', 'danger')
        
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    Base.metadata.create_all(engine)
    
    # Initialize the training status file on start up if it doesn't exist
    if not os.path.exists(TRAINING_PROGRESS_FILE):
        update_training_status(100, 'Ready', False)
    
    # Do NOT hardcode debug=True for production. Use FLASK_DEBUG env var for development.
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('1', 'true', 'yes')
    app.run(debug=debug_mode, host='0.0.0.0')
