---
layout: post
title: "Matrix Decompositions Explained - Part I: Eigen Decomposition"
date: 2025-11-06 10:00:00 +0000
tags: [linear algebra, machine learning]
excerpt: "The Good Cop - Bad Cop Routine."
---

![Omnitrix Header Banner](/assets/images/matrix_decom_eigen/decomp_bad_good_cop_header.png){:class="header-image"}

## Ever wondered why we even need matrix decompositions and what they mean intuitively ? 🤔
In general, decomposition means breaking down something in simpler components.For matrices this is important because it can help us break matrix into simpler building blocks, that tell its story as an object - its *hidden structure*.

***Not "ingredients" but valid representations**

Just to be clear it's not literally breaking it into "ingredients" that originally created the matrix, but mathematically valid ways that can represent the same matrix in terms of simpler pieces, that can be analysed and manipulated as needed.

There are different ways for doing matrix decomposition but in this part we are going to focus on Eigen decompositions. We are going to investigate matrix - our 'criminal' holding secrets. And our objective is to uncover its hidden structure - `eigenvalues` telling us how much it scales and `eigenvectors` the directions that stay unchanged.

## The Eigen interrogation: Good Cop/Bad Cop Routine 👮

We are going to make matrix confess its natural directions and scaling powers, with good cop and bad cop routine:

* **Local View (Good Cop)**: test one vector at a time to see if it's a special direction the matrix only scales.
* **Global View (Bad Cop)**: force the matrix to confess and tell us about its - all eigenvalues and eigenvectors at once - by doing `Matrix Diagonalization`.

Before interrogating our `criminal` matrix, let me tell you - its *symmetric* in nature means he will not lie, though it can hold information. Its up to cops to get all the information out.

<p align="center">
  <img src="/assets/images/matrix_decom_eigen/matrix_decom_eigen_1.png" alt="cops with vector" width="650">
</p>
<p align="center"><em>Eigen-decomposition: Good Cop (left panel) tests one vector at a time, Bad Cop (right panel) makes the matrix confess everything.</em></p>


## Local View
Let us take a candidate vector to test but make sure its from the **same space as the martrix**. Ah! a important point we have stumbled upon and needs to be cleared before we move on forward. 

**Why the "same space" matters?** A matrix $N$ of size $m \times m$ (**a square matrix**) always maps vectors 
from $\mathbb{R}^m$ back into that same space $\mathbb{R}^m$.

> *For eigenvalue/eigenvector testing we need Av=λv, which means both sides of the equation must "live" in the same dimension.*

If the vector came from a different space, the outputs will not be comparable and that is why the probe vector (our candidate vector) should belong to the same vector space and satisfy the following equation:

$$
\underbrace{A}_{\text{Matrix (the criminal)}} 
\; 
\underbrace{\mathbf{v}}_{\text{Candidate vector (from same space)}} 
= 
\underbrace{\lambda}_{\text{Eigenvalue (stretch or shrink factor)}} 
\; 
\underbrace{\mathbf{v}}_{\text{Same vector direction}}
$$


So according to our story think of it this way - 

When the Good Cop brings in a candidate vector for questioning, it must belong to the same 'criminal background' as the matrix. A random innocent bystander (a vector from the wrong space) can't reveal anything about the matrix's secrets. Only those who are part of the same world - the same vector space where the matrix operates - can expose whether the matrix preserves their direction or twists them around.

<figure style="text-align:center;">
  <div style="display:flex; justify-content:center; gap:10px;">
    <img src="/assets/images/matrix_decom_eigen/matrix_decom_eigen_2.png" alt="Image 1" width="45%">
    <img src="/assets/images/matrix_decom_eigen/matrix_decom_eigen_3.png" alt="Image 2" width="45%">
  </div>
  <figcaption><em>Same Space Matters : "Wrong space - wrong suspect."(left panel). "Interrogations only work when they're from the same vector space." (right panel).</em></figcaption>
</figure>

Lets try to understand the behaviour of our simple symmetric matrix when its introduced with different candidate vectors:

```python
import numpy as np

A = np.array([[2, 0],
              [0, -3]])
```
Before I mentioned that our symmetric criminal is truthful - it never twists or shears vectors. It only stretches, shrinks or flips (vector direction is same just flipped in opposite way).

Moving on we bring our three suspects from the same vector space:
* Candidate 1: v1 = [1, 0] -> lies along the x-axis.
* Candidate 2: v2 = [0, 1] -> lies along the y-axis.
* Candidate 3: v3 = [1, 1] -> not algined with any of the matrix's axes.

![Matrix transforming candidate vectors](/assets/images/matrix_decom_eigen/matrix_decom_eigen_4.png)

We observe the behaviour a **symmetric matrix** exhibits under the local view -

> * **Scaling**  
> The matrix keeps the directions but changes the length
> * **Flipping**
> The matrix just flips the vector but direction remains same (negative eiegenvalue)
> * **Direction change** 
> If the matrix changes the direction of the vector, its **not an eigenvector**.


Our Good Cop is doing great job, but its getting cumbersome, progressing slowly - one vector at a time, though he is also discovering how our *symmetric criminal matrix* behaves but its exhausting. From the far end of interoggation room, the Bad Cop has been watching, his patience is burning and suddenly he snaps.


> “Enough of this one-vector nonsense! We’re doing this MY way. Reveal EVERYTHING. All directions and secrets. Right now.”

## Global View

The Bad Cop demands for spilling out the entire structure of the matrix at once - `Diagonalization`

Instead of asking:
> "How do you treat this vector?""

he asks:
> *"Along which directions do you **ALWAYS** stretch or **ALWAYS** flip, no matter who you transform?"*

$$
A \;=\; 
\underbrace{Q}_{\text{eigenvectors}}
\;
\underbrace{\Lambda}_{\text{eigenvalues}}
\;
\underbrace{Q^{\mathsf{T}}}_{\text{transpose of }Q}
$$

This is **diagonalization**, the global view, the full confession and the moment the entire behaviour of the matrix becomes clear at once. 

Before a step further into diagonalization, lets first quickly touch upon the idea of **a change of basis**.
Sometimes a matrix looks complicated in our standard coordinate system, but if we use the **same transformation** in a difference coordinate system, the things become much simpler to understand.
Thats the idea behind using a **similarity transformation** - it let's us describe the same matrix in an new basis where its behaviour is easier to interpret and work with.

Relating this concept back to our story: as the interrogation continues, the Good Cop slightly changes the environment — he walks in with coffee and snacks, hoping to make the criminal matrix a bit more comfortable.
This small shift is our metaphor for a change of basis:
sometimes simply changing the perspective or the coordinate system is enough to make the matrix speak more clearly. 

So *diagonalization is a special case of similarity transformations* and below show the equation breakdown:

$$
A \;=\; 
\underbrace{P}_{\text{columns: eigenvectors}}
\;
\underbrace{D}_{\text{diagonal matrix of eigenvalues}}
\;
\underbrace{P^{-1}}_{\text{change-of-basis back}}
$$

Lets take another example and will make it confess through digonalization.
```python
import numpy as np

A = np.array([[3, 0],
              [0, 0.5]])
```
on solving the it, we get 
```python
P = np.array([[1, 0], 
        [0, 1]])

D = np.array([[3, 0],
            [0, 0.5]])
```

For easier understanding, the criminal matrix taken is simple, and notice the diagonal eigen values matrix is same as our criminal matrix. Now that the Bad Cop has extracted the full confession,
the Good Cop returns to verify the behavior using two test vectors:
```python
vector 1 = np.array([1, 1])

vector 2 = np.array([-1, 1])
```
Once the transformation is applied to them below we can how its gets transformed 
<p align="center">
  <img src="/assets/images/matrix_decom_eigen/matrix_decomp_global_view.png" alt="cops with vector" width="650">
</p>
<p align="center"><em>Global-view: input vector transformation.</em></p>

All code, diagonalization steps, and visualizations are available in the following notebook: [Matrix Decompositions Notebook](https://github.com/luaGeeko/the-storyverse-journal/blob/main/Matrix_Decompositions.ipynb)

So far our Cops did a great job in taking out information from a **well-behaved criminal matrix**- a neat, square one that confessed cleanly through eigenvalues and eigenvectors...**but real criminals aren't always square.** 

So what if the matrix isn't square at all ?
What if it holds **hidden behaviour that eigen decomposition alone can't expose?**

<p align="center">
  <img src="/assets/images/matrix_decom_eigen/eigen_decom_svd_init.png" alt="cops with vector" width="650">
</p>
<p align="center"><em>Next Case: Singular Value Decomposition.</em></p>

In the next part, we need to upgrade our investigation, pushing through will be needed for breaking the criminal. We are going to look into **Singular Value Decomposition(SVD)**- think it as tool our cop detectives are going to use to make criminal confess that refused to be square, simple or honest.

Stay tuned as we open the next case file and bring the entire SVD task force into the interrogation room.



© 2025 Shruti Verma

If this article helped you, feel free to cite or share it — a small mention goes a long way.
I’d also love to hear your thoughts, suggestions, or feedback.

You can reach me here:
• [Email](hop2work@gmail.com)
• [LinkedIn](https://www.linkedin.com/in/shruti31)