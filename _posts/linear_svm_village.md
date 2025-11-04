
# "The untold story of SVM: A Village Divided (Part 1)"

>*This post reimagines the mechanics of Support Vector Machines (SVM) through a story-driven metaphor of a village divided. Beyond the narrative, we dive into the intuition behind linear SVMs — how they construct a hyperplane that maximizes the geometric margin between two classes. We explain the roles of support vectors, the functional margin, and how the direction vector w governs the orientation and classification confidence. Whether you’re a beginner or brushing up on fundamentals, this piece offers a conceptual bridge between storytelling and math.*

Once upon a time, there was a village called Ascots, a peaceful village where two groups Orange Ascots and Blue Ascots lived happily. But on one tragic day, a fight broke out and the disagreement sparked a tension that couldn't be resolved. They cannot relocate to new place, bloodshed to be avoided and above all the peace had to be restored.

Thus, a decision was made by the village’s elder Ascot:

A **boundary must be drawn** — a fence that would separate the two groups as much as possible to avoid any future conflicts.

## Not Just Distance, But Confidence

One question now troubled the elder:

> ***How** should this fence be designed, and **where** exactly should it be built?*

Unsure of the answer, he sought the help of a wise problem solver of the land — **SVM**. Upon accepting the challenge, SVM went for a walk in the village thinking deeply:

> *“If I place a fence, how close will the nearest house be on either side?”*

This thought led to the core idea from a simple concept called the geometric margin :

* the actual distance from the fence (hyperplane) to the closest data point (house).
* and we want to maximize this margin ➕.

But SVM wasn’t only thinking about **distance**. *He also wanted to be **confident** in placing each villager’s house on the correct side of the boundary.*

* Positive ✅ if they were on the right side of the fence
* Negative ❌ if not

This is called the functional margin:

* it’s not the physical distance but a **confidence score** — *how well the data point (house) was classified and how far (in **sign and scale**) it is from the decision boundary.*

![SVM margin illustration](/assets/images/linear_svm/svm_vilage_github1.png)

SVM realized to build the best fence, he had to consider both **score** and the **distance**. So he worked out on the plan and visited the houses that matter the most — the *ones closest to the boundary*, the ones that would support his decision. He also called upon the elder to discuss his strategy.

## The Support Vectors
Ascot the Elder, wanted to understand how the fence was drawn and so SVM explains -

* There could be **multiple fences** (decision boundaries) between two groups. 
* But the **optimal** one he found, makes sure the **distance between the fence and the house of villagers placed on either side of the line is maximum**. That is why he met some of the villagers, calling them — support vectors. The Elder exclaimed:

> *“Ah, so the fence is placed based on the closest villagers to ensure the largest gap between the two groups and actually define where the fence goes!”*

![We need Support Vector](/assets/images/linear_svm/svm_village_github2.png)

## The Equation Behind the Fence
SVM now explains about the functional margin equation.

* It represents a *hyperplane* in geometry — a *flat surface* (like a line in 2D) that separates the space

The *fence isn’t just a line*:

* it has a **direction** (which way it’s tilted) ↖️
* and a **position** (how high or low it sits) ↕️

This is what is called the **functional margin**. It doesn’t just tell us *‘how far’* — it tells us *‘how confident’* we are that each villager is on the correct side. It uses:

* the **direction vector** (w) ➡️.
* and a **bias term (b)** to describe the fence fully.
* The **direction comes from w**: it tells which way the fence tilts.
* The **position is adjusted by b**: it shifts the fence up or down.


$$
w^T x + b = 0
$$

<figure style="text-align:center;">
  <img src="/assets/images/linear_svm/svm_village_github4.png" alt="Hyperplane Equation xkcd" width="500">
  <figcaption><em>Hyperplane Equation Guides.</em></figcaption>
</figure>


## 💬 SVM continues to explain:
> *“Imagine each villager’s home is a point with two coordinates (x₁, x₂).
When I multiply that by a direction vector (w₁, w₂), I’m checking how well the house aligns with that direction.
So now, if the result is positive as per the hyperplane equation, they are to be placed on the positive class — otherwise on the negative class side.”*

Elder looked a bit confused 🤔

> *“Hmm… makes sense, but how do you decide which side is for positive class and which is negative?”*


$$
y =
\begin{cases}
+1, & \text{if } w^T x + b > 0 \\
-1, & \text{if } w^T x + b < 0
\end{cases}
$$

<figure style="text-align:center;">
  <img src="/assets/images/linear_svm/svm_village_github5.png" alt="Door fence xkcd" width="500">
  <figcaption><em>Door in the fence.</em></figcaption>
</figure>

SVM explained, with simple analogy of **door-in-the-fence**, *“Imagine this fence has door in it and the door swings open in the direction of **w**. The side it opens towards is the positive class. The side it swings away from is the negative class.”*

Lets try to understand concepts till here by taking small data and applying linear svm using scikit-learn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# lets take some sample points for 2 classes
X = np.array([
    [2, 2],  # Class +1
    [2, 3],
    [3, 2],
    [6, 5],  # Class -1
    [7, 8],
    [8, 6]
])

# define true labels
y = np.array([1, 1, 1, -1, -1, -1])

clf = svm.SVC(kernel='linear')
# fit to our sample data
clf.fit(X, y)

# extracting w and b
w = clf.coef_[0]
b = clf.intercept_[0]

```


Here we get **w: [-0.33333333 -0.33333333]**, where w[0] corresponds to x-axis (think of 1st feature) and w[1] corresponds to y-axis (2nd feature). Next let's try plotting the points with decision boundary and margin lines.

```python

plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='orange', label='Class +1 (Orange Ascots)', edgecolors='k')
plt.scatter(X[y==-1][:, 0], X[y==-1][:, 1], color='blue', label='Class -1 (Blue Ascots)', edgecolors='k')

# for decision boundary - smooth continuous line lets take points from linspace
x_vals = np.linspace(0, 10, 100)
# compute decsion boundary values
y_vals = -(w[0] * x_vals + b) / w[1]
# compute margin distance from descision boundary to support vectors
margin = 1 / np.sqrt(np.sum(w ** 2))
# we will margin on both the sides 
y_margin_pos = y_vals + margin
y_margin_neg = y_vals - margin

# now lets plot them
plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')
plt.plot(x_vals, y_margin_pos, 'k--', label='Margin (+1)')
plt.plot(x_vals, y_margin_neg, 'k--', label='Margin (-1)')

# legend and labels
plt.legend()
plt.title("Linear SVM with Decision Boundary and Margins")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

```

Now SVM wants us to try out a new point [2.5, 2.5] and see to what class it gets assigned to.

```python
# lets test a new point
test_point = np.array([2.5, 2.5])
# our decision boundary equation
svm_score = np.dot(w, test_point) + b
print(f"Functional margin: {svm_score}")
```

The gives out svm score as 1 and gets class output of +1.

Elder asks *“Oh, interesting — that makes sense now! But when you explained the direction of w with the door-in-the-fence analogy, what if **w** points in the opposite direction? Does that mean the door opens the other way — and the classes get switched?”*

SVM replies *“Exactly! When w points in the opposite direction, the decision boundary stays the same, but the classification flips — what was once the positive side becomes negative, and vice versa.” The below plot shows how flipping w and b changes **classification results**, even though the geometric boundary doesn't move.*

![Decision Boundary w and b](/assets/images/linear_svm/svm_village_github6.png)

You can find the proper code for the above here https://github.com/luaGeeko/MediumMLToons/blob/main/SVM_tutorial.ipynb.

Now wrapping up Part 1, where we laid the foundation by understanding the **functional margin** and **geometric margin** — two crucial concepts that help us measure how well our Support Vector Machine (SVM) separates classes with maximum confidence. In the next part, we’ll take this a step further and explore how to **optimize** the decision boundary to, introduce the role of constraints in this optimization, and uncover the difference between **hard and soft margins** — essential ideas in SVMs.

© 2025 Shruti Verma. All rights reserved.

This blog and its contents are protected under the Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International License.
See [LICENSE](LICENSE) for details.
