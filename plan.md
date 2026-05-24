Probably we can branch depending on uncertainty instead of signal-to-noise ratio

Another thing is that we want the contrast to be high if two sibling leaves lead to different results. They should have high mean and low variance (probably high variance?)

We may train on mean and branch on variance.


We want correct / incorrect clusters to have low variance, and their parent should also have low variance

Parents with alternating correct / incorrect results should have high variance, and their siblings should also have high variance.

