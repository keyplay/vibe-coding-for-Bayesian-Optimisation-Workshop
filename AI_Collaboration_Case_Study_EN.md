# I paired with an AI for a bioprocess optimisation hackathon

Last week I entered a hackathon. The task: use Bayesian optimisation to tune a simulated CHO cell bioreactor, maximising protein titre within a virtual budget of 15,000. Five input variables (temperature, pH, three feed concentrations F1/F2/F3) and three fidelity levels. The cheap 3L reactor costs 10 per run. The expensive 15L reactor costs 2,100.

I used an AI coding assistant the entire time. Eighteen rounds of instructions later, the AI wrote most of the code, but the strategy came out of arguments between us. This is what happened.

## From sklearn to BOTorch: the easy part

The AI started with a sklearn Gaussian process implementation. It worked. I asked if we could switch to BOTorch, and it offered two options: pure BOTorch with KnowledgeGradient, or a hybrid approach using SingleTaskMultiFidelityGP with our own staged budget strategy. I picked the hybrid. I wanted to keep control over how the budget got spent.

The framework migration went fast. SingleTaskMultiFidelityGP, qLogNoisyExpectedImprovement, FixedFeatureAcquisitionFunction. These APIs would have taken me half a day to wire up from documentation alone. The AI had them connected in minutes.

But running code is not the same as correct code.

## The real problem: ghost variables

The low-fidelity simulation (Fid 0) models a 3L reactor with a single feed at t=60 minutes, using F2. F1 and F3 do not exist at this fidelity. The simulator accepts those values but ignores them internally.

Medium and high fidelity (Fid 1/2) are different. All three feeds are active: F1 at t=40, F2 at t=80, F3 at t=120.

The AI's seed migration strategy was to copy the best Fid 0 solution vector directly into Fid 1 as a starting point. Mathematically fine, it is just a six-dimensional vector. Physically, a disaster. F1 and F3 in Fid 0 are junk data, either zeros or meaningless values from Sobol initialisation. The AI carried that junk into the next stage as if it were useful information.

A single Fid 1 experiment costs 575. Starting the search from a bad initial point means wasting several expensive runs just to recover. Thousands of budget units gone.

I asked one question: "Does the low-fidelity experiment only use one feed variable?" The AI admitted it had not considered this.

## Physical intuition vs. full randomisation

The first fix was to randomise F1 and F3 across their full range (0 to 50) during seed migration. Safe, but I thought we could do better.

I looked at the feeding times. Fid 0 feeds at t=60. Fid 1 feeds at t=40 and t=80.

60 sits right between 40 and 80. Biological reactions are continuous processes. Cells do not suddenly change their metabolism because you switched reactors. The optimal feed amount at t=60 is probably related to the optimal amounts at t=40 and t=80.

I told the AI: stop randomising everything. Use the Fid 0 F2 value as an anchor and search nearby.

The AI suggested a perturbation range of plus or minus 15. I thought about it and manually changed it to plus or minus 5.

Why? Because at 575 per experiment, you cannot afford to spray and pray. And the continuity of biological kinetics is stronger than a range of 15 implies. Twenty minutes between t=40 and t=60 is not a long time for cell metabolism to shift.

I later added the same anchor logic for F2 (plus or minus 5) and used a wider range for F3 (plus or minus 10, since t=120 is furthest from t=60 and correlation is likely weaker).

## Five iterations to get the migration right

The seed migration strategy went through five versions:

- v0: Copy everything. F1/F3 are junk, inherited directly.
- v1: Randomise F1/F3 across full range. Safe but inefficient.
- v2: Anchor F1 at plus or minus 15 from Fid 0's F2, randomise F2/F3. Score actually dropped (73.88 to 68.28).
- v3: Anchor both F1 and F2, randomise F3.
- v4: I manually tightened to F1/F2 plus or minus 5, F3 plus or minus 10.
- v5: Added small perturbations to temperature (plus or minus 1 degree C) and pH (plus or minus 0.1).

The interesting part is v2 performing worse than v1. Anchoring only F1 while leaving F2 fully random meant we threw away the t=60 information at the t=80 position. Once we anchored both F1 and F2, performance recovered. Under a tight budget, how thoroughly you use available information determines search efficiency more than anything else.

## What I learned from this partnership

### The AI does not understand "why"

The AI was best at implementation. BOTorch API calls, GP fitting, acquisition function configuration. Fast, accurate work. What it could not do was equally clear: it never asked whether F1 matters at low fidelity, never looked at the timeline and wondered about the relationship between t=60 and t=40, never got cautious because a single run costs 575.

Those are domain knowledge and engineering judgment. Still a human job.

### Questions were worth more than code

Looking back at the log, the things that moved the project forward were not code blocks. They were questions:

- "Does the low-fidelity experiment only use one feed variable?" (found the ghost variables)
- "60 is close to both 40 and 80, can we use that?" (injected physical intuition)
- "Can we add small perturbations to T and pH during migration?" (avoided local optima)

Each question came from understanding the physical process, not the codebase. The AI could implement any idea in seconds. It could not generate these ideas on its own.

### Obedience is a liability

The AI never pushed back. Tell it to copy invalid variables across fidelities, it copies. Tell it to use a perturbation range of 15, it uses 15. It will not say "hold on, this variable is inactive at low fidelity, are you sure you want to carry it over?"

In a hackathon where time is short, this kind of compliance is dangerous. Code that runs feels like code that works. But running and correct can be thousands of budget units apart.

### Iteration beats getting it right the first time

The migration strategy took five versions to stabilise. Not because we were slow, but because multi-fidelity optimisation is full of details that are not obvious upfront. The semantic mismatch between Fid 0 and Fid 1 variables, the inconsistent feeding times, the nonlinear budget allocation. No first draft was going to catch all of that.

The upside of working with an AI is that iteration is cheap. Spot a problem, describe it, the AI rewrites the code, run a test. A few minutes per cycle. Doing this by hand, just the BOTorch API alone would have eaten an afternoon.

## Final thought

This hackathon gave me a concrete picture of what human-AI collaboration looks like in practice. The AI is a partner with infinite typing speed and every API doc memorised. It does not know what you are trying to do, and it does not know when to stop and think.

The final solution (BOTorch, time-correlated seed migration, manually tuned perturbation ranges) came out of 18 rounds of conversation. Left to itself, the AI would probably have delivered something that compiles, makes internal sense, blows the budget, and searches in the wrong direction. Left to myself, I would probably still be reading BOTorch documentation.

Neither of us could have done it alone. Together, it turned out fine.
