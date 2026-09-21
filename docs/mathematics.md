# Mathematics

## Forward Kinematics

For a 2R planar arm with link lengths \(L_1, L_2\):

\[
x = L_1 \cos\theta_1 + L_2 \cos(\theta_1+\theta_2)
\]
\[
y = L_1 \sin\theta_1 + L_2 \sin(\theta_1+\theta_2)
\]

## Workspace

\[
|L_1 - L_2| \le \sqrt{x^2+y^2} \le L_1 + L_2
\]

## Analytical IK

Law of cosines:

\[
\cos\theta_2 = \frac{x^2+y^2 - L_1^2 - L_2^2}{2 L_1 L_2}
\]

Two solutions for \(\theta_2\) (elbow-up / elbow-down). \(\theta_1\) follows from
an adjusted two-argument arctangent.

## Singularities

The Jacobian determinant of the 2R arm is proportional to \(\sin\theta_2\).
Configurations with \(\theta_2 \approx 0\) or \(\pm\pi\) are singular
(fully stretched or folded).

## Position Error

\[
e = \sqrt{(x_{\mathrm{pred}}-x_{\mathrm{target}})^2 + (y_{\mathrm{pred}}-y_{\mathrm{target}})^2}
\]
