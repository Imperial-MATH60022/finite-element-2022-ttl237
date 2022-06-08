"""Solve a model helmholtz problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import cos, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Helmholtz problem given
    the function space in which to solve and the right hand side
    function."""


    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)
    


   
    #raise NotImplementedError
    
    fe = fs.element
    mesh = fs.mesh   
    

    # Constructing my QuadraturreRule
    Q = gauss_quadrature(fe.cell,2*fe.degree)

    # Tabulating the basis function at each quadrature point
    phi = (fe.tabulate(Q.points))
    phigrad = fe.tabulate(Q.points,grad=True)
    #Assembling the RHS
    v = 0
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes= fs.cell_nodes[c, :]
        # Compute the change of coordinates
        J = mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))
        v = np.einsum( " qi , k , qk , q ->  i"  , phi , f.values[nodes], phi, Q.weights)
        v *=  detJ
        l[nodes] += v
    #print(l.shape)
    #Assembling the LHS:

    # phi = phi.T
    # phigrad = np.transpose(phigrad,axes=(1,0,2))
    b = 0
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes= fs.cell_nodes[c, :]
        # Compute the change of coordinates
        J = mesh.jacobian(c)
        JinvT = (np.linalg.inv(J)).T
        detJ = np.abs(np.linalg.det(J))
        b = np.einsum("ba,qib,ya,qiy,qi,qj,q->ij",JinvT,phigrad,JinvT,phigrad,phi,phi,Q.weights)
        b*=detJ
        A[np.ix_(nodes,nodes)] += b
            
    # Create an appropriate (complete) quadrature rule.

    # Tabulate the basis functions and their gradients at the quadrature points.

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    

    # Now loop over all the cells and assemble A and l

    return A, l




def solve_helmholtz(degree, resolution, analytic=False, return_error=False):
    """Solve a model Helmholtz problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: cos(4*pi*x[0])*x[1]**2*(1.-x[1])**2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: ((16*pi**2 + 1)*(x[1] - 1)**2*x[1]**2 - 12*x[1]**2 + 12*x[1] - 2) *
                  cos(4*pi*x[0]))

    # Assemble the finite element system.
    A, l = assemble(fs, f)

    # Create the function to hold the solution.
    u = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Helmholtz problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_helmholtz(degree, resolution, analytic, plot_error)

    u.plot()
