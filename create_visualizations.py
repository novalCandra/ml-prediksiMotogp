import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
colors = sns.color_palette("husl", 8)

# Load predictions
predictions = pd.read_csv('data/championship_predictions_2025.csv')
top_riders = predictions[predictions['Championship Probability'] > 0].head(10)

# 1. Championship Probability Comparison
fig, ax = plt.subplots(figsize=(12, 6))
riders = top_riders['Rider'].values
probs = top_riders['Championship Probability'].values * 100

bars = ax.barh(riders, probs, color=['#FF6B6B' if i == 0 else '#4ECDC4' if i == 1 else '#45B7D1' if i == 2 else '#96CEB4' for i in range(len(riders))])

# Add value labels
for i, (bar, prob) in enumerate(zip(bars, probs)):
    ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold')

ax.set_xlabel('Championship Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('2025 MotoGP Championship Predictions\nTop 10 Contenders', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, max(probs) * 1.15)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('data/championship_probability.png', dpi=300, bbox_inches='tight')
print("✓ Saved: championship_probability.png")
plt.close()

# 2. Top 3 Riders - Detailed Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Top 3 Predicted Champions - Detailed Analysis', fontsize=16, fontweight='bold', y=1.00)

top_3 = predictions.head(3)

# 2a. Points Comparison
ax = axes[0, 0]
riders_names = top_3['Rider'].values
points = top_3['Points'].values
colors_top3 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(riders_names, points, color=colors_top3, edgecolor='black', linewidth=1.5)
for bar, point in zip(bars, points):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(point)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Points', fontweight='bold')
ax.set_title('Total Points (2025 Season)', fontweight='bold')
ax.set_ylim(0, max(points) * 1.15)
ax.grid(axis='y', alpha=0.3)

# 2b. Podiums Comparison
ax = axes[0, 1]
podiums = top_3['Podiums'].values
bars = ax.bar(riders_names, podiums, color=colors_top3, edgecolor='black', linewidth=1.5)
for bar, podium in zip(bars, podiums):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(podium)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Podiums', fontweight='bold')
ax.set_title('Podium Finishes', fontweight='bold')
ax.set_ylim(0, max(podiums) * 1.5)
ax.grid(axis='y', alpha=0.3)

# 2c. Wins Comparison
ax = axes[1, 0]
wins = top_3['Wins'].values
bars = ax.bar(riders_names, wins, color=colors_top3, edgecolor='black', linewidth=1.5)
for bar, win in zip(bars, wins):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(win)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Wins', fontweight='bold')
ax.set_title('Race Wins', fontweight='bold')
ax.set_ylim(0, max(wins) * 1.5 if max(wins) > 0 else 1.5)
ax.grid(axis='y', alpha=0.3)

# 2d. Championship Probability
ax = axes[1, 1]
probs_top3 = top_3['Championship Probability'].values * 100
bars = ax.bar(riders_names, probs_top3, color=colors_top3, edgecolor='black', linewidth=1.5)
for bar, prob in zip(bars, probs_top3):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Probability (%)', fontweight='bold')
ax.set_title('Championship Probability', fontweight='bold')
ax.set_ylim(0, max(probs_top3) * 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/top3_detailed_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top3_detailed_comparison.png")
plt.close()

# 3. Team Distribution
fig, ax = plt.subplots(figsize=(12, 6))
team_counts = predictions['Team'].value_counts().head(8)
colors_teams = sns.color_palette("Set2", len(team_counts))
bars = ax.barh(team_counts.index, team_counts.values, color=colors_teams, edgecolor='black', linewidth=1.5)

for bar, count in zip(bars, team_counts.values):
    ax.text(count + 0.1, bar.get_y() + bar.get_height()/2.,
            f'{int(count)}', va='center', fontweight='bold')

ax.set_xlabel('Number of Riders', fontweight='bold')
ax.set_title('2025 MotoGP Teams - Rider Distribution', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('data/team_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: team_distribution.png")
plt.close()

# 4. Motorcycle Distribution
fig, ax = plt.subplots(figsize=(12, 6))
moto_counts = predictions['Motorcycle'].value_counts()
colors_moto = sns.color_palette("husl", len(moto_counts))
wedges, texts, autotexts = ax.pie(moto_counts.values, labels=moto_counts.index, autopct='%1.1f%%',
                                    colors=colors_moto, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})

ax.set_title('2025 MotoGP Motorcycle Distribution', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('data/motorcycle_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: motorcycle_distribution.png")
plt.close()

# 5. Points Per Race vs Championship Probability
fig, ax = plt.subplots(figsize=(12, 7))

# Filter riders with data
riders_with_data = predictions[predictions['PPR'] > 0].copy()

scatter = ax.scatter(riders_with_data['PPR'], riders_with_data['Championship Probability'] * 100,
                     s=riders_with_data['Points'] * 3 + 100, alpha=0.6, c=riders_with_data['Championship Probability'],
                     cmap='RdYlGn', edgecolors='black', linewidth=1.5)

# Annotate top riders
for idx, row in riders_with_data.head(5).iterrows():
    ax.annotate(row['Rider'], (row['PPR'], row['Championship Probability'] * 100),
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.set_xlabel('Points Per Race', fontsize=12, fontweight='bold')
ax.set_ylabel('Championship Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('Championship Probability vs Points Per Race\n(Bubble size = Total Points)', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Championship Probability', fontweight='bold')

plt.tight_layout()
plt.savefig('data/ppr_vs_probability.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ppr_vs_probability.png")
plt.close()

# 6. Country Distribution
fig, ax = plt.subplots(figsize=(12, 6))
country_counts = predictions['Country'].value_counts()
colors_country = sns.color_palette("Set3", len(country_counts))
bars = ax.barh(country_counts.index, country_counts.values, color=colors_country, edgecolor='black', linewidth=1.5)

for bar, count in zip(bars, country_counts.values):
    ax.text(count + 0.1, bar.get_y() + bar.get_height()/2.,
            f'{int(count)}', va='center', fontweight='bold')

ax.set_xlabel('Number of Riders', fontweight='bold')
ax.set_title('2025 MotoGP Riders by Country', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('data/country_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: country_distribution.png")
plt.close()

# 7. Performance Metrics Heatmap (Top 10)
fig, ax = plt.subplots(figsize=(12, 8))

top_10 = predictions.head(10).copy()
metrics = ['Points', 'Wins', 'Podiums', 'Poles', 'Fastest Laps', 'PPR']

# Normalize metrics for heatmap
heatmap_data = top_10[metrics].copy()
for col in metrics:
    max_val = heatmap_data[col].max()
    if max_val > 0:
        heatmap_data[col] = heatmap_data[col] / max_val

sns.heatmap(heatmap_data.T, annot=top_10[metrics].T.values, fmt='.1f', cmap='YlOrRd',
            xticklabels=top_10['Rider'].values, yticklabels=metrics, cbar_kws={'label': 'Normalized Score'},
            linewidths=1, linecolor='black', ax=ax)

ax.set_title('Top 10 Riders - Performance Metrics Heatmap\n(Normalized Scores)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Rider', fontweight='bold')
ax.set_ylabel('Metric', fontweight='bold')

plt.tight_layout()
plt.savefig('data/performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_heatmap.png")
plt.close()

print("\n" + "="*60)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
print("="*60)
print("\nGenerated Files:")
print("  1. championship_probability.png - Top 10 contenders")
print("  2. top3_detailed_comparison.png - Detailed analysis")
print("  3. team_distribution.png - Teams in 2025")
print("  4. motorcycle_distribution.png - Bike distribution")
print("  5. ppr_vs_probability.png - Correlation analysis")
print("  6. country_distribution.png - Riders by country")
print("  7. performance_heatmap.png - Metrics comparison")
