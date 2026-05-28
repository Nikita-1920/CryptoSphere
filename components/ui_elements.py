import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def render_metric_card(title, value, delta, is_positive, icon=""):
    delta_class = "positive" if is_positive else "negative"
    delta_symbol = "▲" if is_positive else "▼"
    
    html = f"""
    <div class="glass-container" style="margin-bottom: 1rem;">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta {delta_class}">{delta_symbol} {delta}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def plot_candlestick(df, symbol, theme="Dark", show_bb=False, show_macd=False, show_rsi=False, target_price=None):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    text_color = '#ffffff' if theme == 'Dark' else '#0f172a'
    
    # Determine subplot structure
    num_subplots = 1 + int(show_macd) + int(show_rsi)
    row_heights = [0.6] if num_subplots == 1 else ([0.5] + [0.5/(num_subplots-1)]*(num_subplots-1))
    
    fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)
    
    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='#10b981', 
                decreasing_line_color='#ef4444',
                name='Price'), row=1, col=1)
                
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='#3b82f6', width=1.5), name='SMA 20'), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#8b5cf6', width=1.5), name='SMA 50'), row=1, col=1)
        
    if show_bb and 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(167, 139, 250, 0.5)', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(167, 139, 250, 0.5)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(167, 139, 250, 0.1)', name='BB Lower'), row=1, col=1)

    if target_price and target_price > 0:
        fig.add_hline(y=target_price, line_dash="dot", line_color="#ec4899", annotation_text=f"Target: {target_price}", annotation_position="top left", row=1, col=1)

    current_row = 2
    # Row 2/3: MACD
    if show_macd and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#3b82f6', width=1.5), name='MACD'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#f59e0b', width=1.5), name='Signal'), row=current_row, col=1)
        colors = ['#10b981' if val >= 0 else '#ef4444' for val in (df['MACD'] - df['MACD_Signal'])]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['MACD_Signal'], marker_color=colors, name='Histogram'), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", title_font=dict(color=text_color), tickfont=dict(color=text_color), row=current_row, col=1)
        current_row += 1

    # Row 2/3: RSI
    if show_rsi and 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='#a855f7', width=1.5), name='RSI'), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.3)" if theme=="Dark" else "rgba(0,0,0,0.3)", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.3)" if theme=="Dark" else "rgba(0,0,0,0.3)", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], title_font=dict(color=text_color), tickfont=dict(color=text_color), row=current_row, col=1)

    # Hide rangeslider for all x-axes and apply text color
    fig.update_xaxes(rangeslider_visible=False, tickfont=dict(color=text_color))
    fig.update_yaxes(title_text='Price', title_font=dict(color=text_color), tickfont=dict(color=text_color), row=1, col=1)

    fig.update_layout(
        title=dict(text=f'{symbol} Price Chart', font=dict(size=22, color=text_color)),
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0),
        height=400 + (num_subplots-1)*200,
        showlegend=False
    )
    return fig

def plot_volatility(df, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    fig = go.Figure()
    if 'Volatility' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], line=dict(color='#f59e0b', width=2), name='Volatility', fill='tozeroy'))
    
    fig.update_layout(
        title=dict(text='Volatility Trend', font=dict(size=22, color='#ffffff' if theme == 'Dark' else '#0f172a')),
        yaxis_title='Volatility',
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_sentiment(df, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    fig = go.Figure()
    if 'Sentiment' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, 
            y=df['Sentiment'],
            marker_color=['#10b981' if val > 0 else '#ef4444' for val in df['Sentiment']],
            name='Daily Sentiment'
        ))
    
    fig.update_layout(
        title=dict(text='News Sentiment Polarity', font=dict(size=22, color='#ffffff' if theme == 'Dark' else '#0f172a')),
        yaxis_title='Sentiment Score',
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_correlation_heatmap(df, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    corr = df[['Close', 'Volume', 'MA_7', 'RSI_14', 'Volatility', 'Momentum', 'Sentiment']].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
    fig.update_layout(
        title=dict(text='Feature Correlation Heatmap', font=dict(size=22, color='#ffffff' if theme == 'Dark' else '#0f172a')),
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def plot_gauge_chart(score, title, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 22, 'color': '#ffffff' if theme == 'Dark' else '#0f172a'}},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "rgba(0,0,0,0.5)" if theme == "Light" else "white"},
            'steps' : [
                {'range': [-1, -0.2], 'color': "#ef4444"},
                {'range': [-0.2, 0.2], 'color': "#94a3b8"},
                {'range': [0.2, 1], 'color': "#10b981"}],
        }
    ))
    fig.update_layout(
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=20, r=20, t=70, b=20)
    )
    return fig

def plot_normalized_comparison(df_dict, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    fig = go.Figure()
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#ec4899', '#14b8a6']
    for i, (coin, df) in enumerate(df_dict.items()):
        if not df.empty and 'Close' in df.columns:
            normalized_close = (df['Close'] / df['Close'].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=normalized_close, 
                mode='lines', 
                name=coin,
                line=dict(width=2, color=colors[i % len(colors)])
            ))
            
    fig.update_layout(
        title=dict(text='Normalized Performance Comparison (Base 100)', font=dict(size=22, color='#ffffff' if theme == 'Dark' else '#0f172a')),
        yaxis_title='Normalized Price',
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified"
    )
    return fig

def plot_portfolio_allocation(portfolio_df, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    text_color = '#ffffff' if theme == 'Dark' else '#0f172a'
    
    # Filter out 0 values for pie chart
    plot_df = portfolio_df[portfolio_df['Value'] > 0]
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Portfolio is empty", showarrow=False, font=dict(size=20, color=text_color))
    else:
        fig = px.pie(
            plot_df, 
            values='Value', 
            names='Asset', 
            hole=0.7,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        # Update text color for pie labels
        fig.update_traces(textfont_color=text_color, textfont_size=13)
        
        # Add total value in center
        total_val = plot_df['Value'].sum()
        fig.add_annotation(
            text=f"<b>Total Value</b><br>${total_val:,.2f}",
            showarrow=False,
            font=dict(size=18, color=text_color)
        )
    
    fig.update_layout(
        title=dict(text='Portfolio Allocation', font=dict(size=22, color=text_color)),
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(font=dict(color=text_color, size=13))
    )
    return fig

def plot_equity_curve(df, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    text_color = '#ffffff' if theme == 'Dark' else '#0f172a'
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Equity Curve
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Equity'],
        line=dict(color='#10b981', width=2),
        name='Equity',
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ), row=1, col=1)
    
    # Drawdown
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Drawdown'] * 100, # Convert to %
        line=dict(color='#ef4444', width=1),
        name='Drawdown %',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.3)'
    ), row=2, col=1)
    
    fig.update_yaxes(title_text='Portfolio Value ($)', title_font=dict(color=text_color), tickfont=dict(color=text_color), row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', title_font=dict(color=text_color), tickfont=dict(color=text_color), row=2, col=1)
    fig.update_xaxes(tickfont=dict(color=text_color))
    
    fig.update_layout(
        title=dict(text='Strategy Equity Curve & Drawdown', font=dict(size=22, color=text_color)),
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0),
        height=500,
        showlegend=False
    )
    return fig

def plot_monte_carlo(df, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    text_color = '#ffffff' if theme == 'Dark' else '#0f172a'
    
    fig = go.Figure()
    
    # 95th Percentile (Optimistic)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['P95 (Optimistic)'],
        line=dict(color='rgba(16, 185, 129, 0.2)', width=0),
        name='95th Percentile',
        showlegend=False
    ))
    
    # 5th Percentile (Pessimistic)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['P5 (Pessimistic)'],
        line=dict(color='rgba(239, 68, 68, 0.2)', width=0),
        name='5th Percentile',
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)',
        showlegend=False
    ))
    
    # 50th Percentile (Expected)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['P50 (Expected)'],
        line=dict(color='#3b82f6', width=2),
        name='Expected Value (P50)'
    ))
    
    fig.update_layout(
        title=dict(text='Monte Carlo Portfolio Projection (30 Days)', font=dict(size=22, color=text_color)),
        yaxis_title='Projected Value ($)',
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color))
    )
    
    fig.update_xaxes(tickfont=dict(color=text_color))
    fig.update_yaxes(tickfont=dict(color=text_color), title_font=dict(color=text_color))
    
    return fig

def plot_fear_greed_gauge(score, title, theme="Dark"):
    t = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 22, 'color': '#ffffff' if theme == 'Dark' else '#0f172a'}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "rgba(0,0,0,0.5)" if theme == "Light" else "white"},
            'steps' : [
                {'range': [0, 25], 'color': "#ef4444"}, # Extreme Fear (Red)
                {'range': [25, 45], 'color': "#f97316"}, # Fear (Orange)
                {'range': [45, 55], 'color': "#94a3b8"}, # Neutral (Gray)
                {'range': [55, 75], 'color': "#84cc16"}, # Greed (Light Green)
                {'range': [75, 100], 'color': "#10b981"} # Extreme Greed (Dark Green)
            ],
        }
    ))
    fig.update_layout(
        template=t,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(l=20, r=20, t=70, b=20)
    )
    return fig
