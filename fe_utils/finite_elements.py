# Cause division to always mean floating point division.
from __future__ import division
from collections import defaultdict
from email.policy import default
from turtle import end_fill
from webbrowser import Elinks
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    # Computation of the Lagrange points
    prescon = []       #points respecting the condition
    if cell.dim == 1:
        for i in range(0,degree+1):
            prescon.append([i/degree])
    elif cell.dim == 2:
        for i in range(0,degree+1):
            for j in range(0,degree+1):
                if 0<=i+j<=degree:
                    prescon.append([j/degree,i/degree])
    else:
        raise NotImplementedError
    return np.array(prescon)

def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    # Computing the Vandermonde Matrix
    points = np.array(points).astype(float)
    if grad == False:
        vonmat = [[] for i in range (len(points))]
        if cell.dim == 1:
            for i in range(0,degree+1):
                xp = i
                for p in range(len(points)):
                    vonmat[p].append(points[p][0]**xp)
        elif cell.dim == 2:
            for i in range(0,degree+1):
                for yp in range(0,i+1):                 #yp is the power for y
                    xp = i-yp                           #xp is the power of the corresponding terms of x
                    for p in range(len(points)):
                        vonmat[p].append(points[p][0]**xp * points[p][1]**yp)
        return np.array(vonmat)
        
        #Computing the gradient of the Vandermone Matrix
    elif grad == True:
        if cell.dim == 1:
            vonmatgrad = [[[0]] for i in range (len(points))]
            for i in range(1,degree+1):
                xp = i
                for p in range(len(points)):
                    vonmatgrad[p].append([xp*(points[p][0]**(xp-1))])
        if cell.dim == 2:
            vonmatgrad = [[] for i in range (len(points))]
            for i in range(0,degree+1):
                for yp in range(0,i+1):
                    xp = i-yp
                    for p in range(len(points)):
                        vonmatgrad[p].append([(xp*(points[p][0]**(xp-1))) * points[p][1]**yp, points[p][0]**xp * (yp*(points[p][1]**(yp-1)))])  

        return np.array(np.nan_to_num(vonmatgrad))



class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        
        if entity_nodes:
                        
        #: ``nodes_per_entity[d]`` is the number of entities
        #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(self.entity_nodes[d][0])
                                            for d in range(self.cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        #Basis Coefficeints
        self.basis_coefs  = np.linalg.inv(vandermonde_matrix(self.cell,self.degree,self.nodes))
        

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def cal_entity_nodes(self,degree):
       
        if self.cell.dim == 2:
            c = len(lagrange_points(self.cell,degree)) - 1
            x = degree+1
            
            d0 = {0:[0],1:[degree],2:[c]} #dict of vertices

            # Calculating points of edges(1,0)
            arr0 = []
            for i in range (1,x-1):
                arr0.append(int((i+1)*x - (i+1)*i/2 - 1))
           
            #Calculating the points of the edges(1,1)
            arr1=[]
            for i in range (1,x-1):
                arr1.append(int(i/2 * (2*x +1 -i)))

            # Calculating for edges(1,2)
            arr2 = []
            for i in range(1,x-1):
                arr2.append(int(i))
            
            d1= {0:arr0, 1:arr1, 2:arr2} # Dictionary of edges

            #Calculating the points in the interior of cell (2,0)
            arr3 = []
            for i in range(0,len(arr1)-1):
                count = len(arr1) - (1+i)
                for j in range(count):
                    xp = arr1[i] + j +1
                    arr3.append(xp) 
            
            entity_nodes_cal ={
            0: d0,
            1: {0:[],1:[],2:[]} if degree <= 1 else d1,
            2: {0:[]} if degree < 3 else {0:arr3}
            }
            return entity_nodes_cal

        elif self.cell.dim == 1:
            c = len(lagrange_points(self.cell,degree)) - 1
            x = degree+1
            d0 = {0:[0],1:[c]} # Dictionary of vertices

            # Calculating for edges(1,0)
            arr2 = []
            for i in range(1,x-1):
                arr2.append(int(i))

            d1= {0:arr2} #Dictionary of edge

            entity_nodes_cal ={
            0: d0,
            1: {0:[]} if degree <= 1 else d1,
            }

            return entity_nodes_cal
        


    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        # Evaluating a set of basis functions at a set of point(Tabulation)
        vx = vandermonde_matrix(self.cell,self.degree,points,grad=grad)
        # Gradient of the Basis Coefficients
        if grad:
            if self.cell.dim == 1:
                e0 = np.array([[1]])
            elif self.cell.dim == 2:
                e0 = np.array([[1,0],[0,1]])
            tabulatematrix = np.einsum("ijk,jl->ilk", vx, self.basis_coefs)
            tabulatematrix = np.einsum("ijk,kl->ijl", tabulatematrix, e0)            
        else:
            tabulatematrix = np.dot(vx,self.basis_coefs)
        return tabulatematrix
        

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        A = [] # List containing the value of "fn"
        for node in self.nodes:
            A.append(fn(node))
        return np.array(A)

       

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """
        self.cell = cell
        self.degree = degree 
        
        # Use lagrange_points to obtain the set of nodes.  Once you
        nodes = lagrange_points(cell, degree)
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        entity_nodes = self.cal_entity_nodes(degree)
        super(LagrangeElement, self).__init__(cell, degree, nodes,entity_nodes=entity_nodes)

