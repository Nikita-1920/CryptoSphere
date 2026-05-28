import pandas as pd
from data.database import get_user_portfolio, save_user_portfolio

def add_to_portfolio_logic(user_id, asset: str, amount: float) -> str:
    if not user_id:
        return "Error: User is not logged in. Tell the user they must log in to use this feature."
        
    df = get_user_portfolio(user_id)
    
    # Check if asset exists
    if asset in df['Asset'].values:
        idx = df.index[df['Asset'] == asset].tolist()[0]
        df.at[idx, 'Amount'] = float(df.at[idx, 'Amount']) + float(amount)
    else:
        new_row = pd.DataFrame([{'Asset': asset, 'Amount': float(amount)}])
        df = pd.concat([df, new_row], ignore_index=True)
        
    save_user_portfolio(user_id, df)
    return f"Successfully added {amount} {asset} to the portfolio."

def get_portfolio_summary_logic(user_id) -> str:
    if not user_id:
        return "Error: User is not logged in. Tell the user they must log in to use this feature."
        
    df = get_user_portfolio(user_id)
    
    if df.empty:
        return "The portfolio is currently empty."
        
    summary = "Current Portfolio Holdings:\n"
    for _, row in df.iterrows():
        if pd.notnull(row['Amount']) and float(row['Amount']) > 0:
            summary += f"- {row['Asset']}: {row['Amount']}\n"
        
    return summary
