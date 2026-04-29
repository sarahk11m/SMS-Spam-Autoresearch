# Failure Log

## Week 3 common failure modes

1. **Small-sample ceiling effects**
   Multiple model variants reached 1.0000 validation macro-F1 on the fallback sample, which makes it hard to tell whether any one improvement is truly meaningful.

2. **Tied metric results**
   Several experiments produced the same top-line score, so keep/discard decisions still require human judgment rather than a purely automatic rule.

3. **Overinterpreting dry runs**
   An apparently better model on the fallback sample may not stay better on the full UCI SMS Spam Collection.

4. **Search-space ambiguity**
   There are many plausible TF-IDF and classifier changes, so the search still needs a disciplined ordering of experiments.

5. **Complexity versus simplicity tradeoffs**
   Some variants improved the score, but not all improvements were equally appealing. A small, same-family change may be easier to justify than a broader model-family swap.

## Week 3 takeaway

The main failure pattern so far is not broken code. It is **evaluation instability caused by a small fallback sample**. The loop works, but the next stage needs a larger corpus to separate real signal from easy wins on a toy setting.
