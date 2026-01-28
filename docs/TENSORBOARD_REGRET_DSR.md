# TensorBoard : Triptyque du Succès (RegretDSR + Gated Policy)

Pour valider que les hyperparamètres d’entropie et la Gated Policy fonctionnent correctement avec RegretDSR, utilisez une vue personnalisée dans TensorBoard avec ces trois graphiques superposés.

## Vue personnalisée

| Graphique | Clé TensorBoard | Ce qu’il valide | Comportement attendu |
|-----------|-----------------|-----------------|----------------------|
| **Envie de trader (Gate)** | `policy/gate_open_ratio` | L’envie de trader | Doit démarrer haut (exploration), baisser, puis remonter par vagues selon les opportunités. |
| **Qualité du signal** | `reward/regret_dsr_raw` ou `rewards/regret_dsr_raw` | La qualité du signal RegretDSR | Doit passer de valeurs négatives (Regret) vers positives ou nulles (Alpha). |
| **Efficacité du sizing** | `policy/action_saturation` | L’efficacité du sizing | Doit être corrélée à la volatilité du marché. Plus le marché bouge, plus l’agent doit être saturé (position tranchée). |

## Clés TensorBoard

- **`policy/gate_open_ratio`** : issu de `model.policy.actor._last_gate_val` (RobustActor / Gated Policy). Part moyenne d’envs où la Gate est ouverte.
- **`reward/regret_dsr_raw`** : alias de `rewards/regret_dsr_raw`. Valeur brute du DSR sur l’Alpha (agent vs buy‑and‑hold), avant normalisation.
- **`policy/action_saturation`** : part des envs avec `|position| > 0.1` (activité / saturation des positions).

## Hyperparamètres associés (Gated Policy / RegretDSR)

| Paramètre | Valeur recommandée | Impact |
|-----------|--------------------|--------|
| `target_entropy` | `-1.5` | Exploration plus fine une fois la Gate ouverte. |
| `log_std_init` | `-1.0` | Bruit initial plus fort pour sortir du cash trap. |
| `learning_starts` | `10_000` | Remplir le replay de Regret avant d’apprendre. |
| `tau` | `0.005` | Le Critic intègre le Regret progressivement. |
| `repulsion_weight` | `0.05` | Poids du Bimodal Prior (repousser du centre). |

EntropyFloorCallback et DynamicEntropyFloorCallback maintiennent α au-dessus d’un seuil et le boostent en cas de saturation trop faible (cash trap).
