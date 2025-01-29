# RocketMoney Prototype

RocketMoney Prototype is a Streamlit-based web application designed to help users manage their subscriptions by connecting to their bank accounts via Plaid. 

## Features

- **User Authentication**: Secure registration and login system with hashed passwords.
- **Plaid Integration**: Connect bank accounts to fetch account details and transactions.
- **Subscription Management**: Identify recurring transactions and allow users to manage their subscriptions.
- **Responsive UI**: Clean and organized interface with custom styling.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**
   Create a `.env` file with:
   ```
   SECRET_KEY=your_secret_key
   PLAID_CLIENT_ID=your_plaid_client_id
   PLAID_SECRET=your_plaid_secret
   DATABASE_URL=sqlite:///app.db
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Directory Structure
```
/RocketMoney
    |-- app.py
    |-- README.md
    |-- requirements.txt
    |-- templates/
        |-- dashboard.html
        |-- login.html
        |-- logout.html
        |-- register.html
        |-- subscriptions.html
    |-- static/
        |-- styles.css
```
