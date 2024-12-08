
### Why do we want Jabocian?


### Compute Jabocian

Six parameters $\{x, y, z, \phi, \theta, \psi\}$

Image plane projection formula: $\tilde{\bold x}_{i,j} = \tilde{\bold K} (\bold T_{WC_i})^{-1} \bar{\bold p_j}$

Where $j$: landmark

Recall that 
$$ \bar{\bold p_j} = \begin{bmatrix}x \\ y \\ z\end{bmatrix} $$

$$ \therefore \frac{\partial \tilde{\bold x}_{i,j}}{\partial \bar{\bold p_j}} = \tilde{\bold K} (\bold T_{WC_i})^{-1} $$

Now we have $\tilde{\bold x}_{i,j} = \tilde{\bold K} \bold T(\phi, \theta, \psi)^{-1} \bar{\bold p_j}$

TESTING


$$
\dot{\tilde{\bold x}} = \tilde{\bold K} (C^{T} \tilde{p} - C^{T} t)
$$

$$
\frac{\partial \tilde{\bold x}}{\partial x} = \tilde{\bold K} (- (C^T)_1)
$$

$$
\frac{\partial \tilde{\bold x}}{\partial y} = \tilde{\bold K} (- (C^T)_2)
$$

$$
\frac{\partial \tilde{\bold x}}{\partial z} = \tilde{\bold K} (- (C^T)_3)
$$

$$
\text{Note} \quad x = \frac{\tilde{\bold x}_1}{\tilde{\bold x}_3}, \quad y = \frac{\tilde{\bold x}_2}{\tilde{\bold x}_3}
$$

$$
\frac{\partial x}{\partial \tilde{\bold x}} =
\begin{bmatrix}
\frac{1}{\tilde{\bold x}_3} & 0 & -\frac{\tilde{\bold x}_1}{\tilde{\bold x}_3^2} \\
0 & \frac{1}{\tilde{\bold x}_3} & -\frac{\tilde{\bold x}_2}{\tilde{\bold x}_3^2}
\end{bmatrix}
$$

$$
\frac{\partial \tilde{\bold x}}{\partial \phi} = \tilde{\bold K} 
\left( 
C_z(\psi) C_y(\theta) \frac{\partial C_x(\phi)}{\partial \phi}  (\tilde{p} - t)^T 
\right)
$$

$$
\frac{\partial \tilde{\bold x}}{\partial \theta} = \tilde{\bold K} 
\left( 
C_z(\psi) \frac{\partial C_y(\theta)}{\partial \theta} C_x(\phi) (\tilde{p} - t)^T 
\right)
$$

$$
\frac{\partial \tilde{\bold x}}{\partial \psi} = \tilde{\bold K} 
\left( 
\frac{\partial C_z(\psi)}{\partial \psi} C_y(\theta) C_x(\phi) (\tilde{p} - t)^T 
\right)
$$

#### Rotation matrices & element-wise derivatives

$$
C_x(\phi) =
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi & \cos\phi
\end{bmatrix}
$$

$$
C_y(\theta) =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{bmatrix}
$$

$$
C_z(\psi) =
\begin{bmatrix}
\cos\psi & -\sin\psi & 0 \\
\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

$$
\frac{\partial C_z}{\partial \psi} =
\begin{bmatrix}
-\sin\psi & -\cos\psi & 0 \\
\cos\psi & -\sin\psi & 0 \\
0 & 0 & 0
\end{bmatrix}
$$
