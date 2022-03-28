## Imports
from functools import reduce
import sympy as sp
import copy
import math as maths
import numpy as np
import time
import cProfile
from sympy.combinatorics.generators import symmetric, Permutation
from sympy import symbols
from fractions import Fraction


## Global functions
def partitions(n, s=1):
    #Author: https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    #Yields the partitions of n
    #s is the smallest value of integers that will appear in the partition
    yield (n,)
    for i in range(s, n // 2 + 1):
        for p in partitions(n - i, i):
            yield p + (i,)


def asc(test_list):
    # Returns true if the list is in ascending order
    return all(i < j for i, j in zip(test_list, test_list[1:]))


def reorder_asc(lst):
    # Returns the list in ascending order
    temp = copy.deepcopy(lst)
    temp.sort()
    return temp


def set_partition(collection):
    # Given a collection of objects, yields its possible partitions into sets
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in set_partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def pos_create_perm(list_1, list_2):
    # Returns the permutation p such that p(list_1) = list_2
    # p acts on positions in the list
    start = Permutation(list_1)
    finish = Permutation(list_2)
    p = finish * start

    return p


def num_create_perm(list_1, list_2):
    # This time, p acts upon numbers
    intermediate = range(len(list_1))
    p_1 = pos_create_perm(list_1, intermediate)
    p_2 = pos_create_perm(intermediate, list_2)

    return p_1 ** (-1) * p_2


def mult_perms(p1, p2):
    # Returns the product permutations p1 and p2
    n = max(p1.size, p2.size)
    p1 = p1 * Permutation(n-1)
    p2 = p2 * Permutation(n-1)
    test = range(n)
    test = p2(p1(test))

    return pos_create_perm(range(p1.size), test)


def sum_list(lst1, lst2):
    # Returns the element-wise sum of two lists of the same length
    assert len(lst1) == len(lst2)
    return [sum(y) for y in zip(lst1, lst2)]


def pos_highest(lst):
    # Returns the position of the greates element in the list
    ''' Source: https://stackoverflow.com/questions/2474015/
    getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list'''
    return max(range(len(lst)), key=lst.__getitem__)


def flatten(lst, ltypes=(list, tuple)):
    # Given an arbitrarily nested list of lists, returns a list
    # Source: http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    ltype = type(lst)
    lst = list(lst)
    i = 0
    while i < len(lst):
        while isinstance(lst[i], ltypes):
            if not lst[i]:
                lst.pop(i)
                i -= 1
                break
            else:
                lst[i:i + 1] = lst[i]
        i += 1
    return ltype(lst)


## YT class
class YT:
    ''' Young tableau represented as a list of lists

        Optional
        Extended
        Description llists, transpose=False, positive=True

        Parameters
        ----------
        llists : list
            List of lists representing the entries of the Young tableau
        transpose=False : boolean
            If True, generates instead the transpose
        positive=True : boolean
            Represents the sign of the tableau

        Attributes
        ----------
        llists : list
            List of lists representing the entries of the Young tableau
        transpose L list
            List of lists representing the transpose of the Young tableau
        sign : boolean
            Sign of the Young tableau
        n : int
            Number of boxes of the Young tableau
        '''

    def __init__(self, llists, transpose=False, positive=True):
        self.llists = llists
        self.transpose = YT.transpose(self)
        self.sign = positive
        self.n = sum([len(llist) for llist in llists])
        if transpose:
            self.llists, self.transpose = self.transpose, self.llists

    def transpose(self):
        # Transposes YT
        new_part = [[] for x in range(len(self.llists[0]))]
        for i in range(len(self.llists)):
            for j in range(len(self.llists[i])):
                new_part[j].append(self.llists[i][j])

        return new_part

    # Printing
    def disp(self, sgn=False):
        # Displays the Young tableau
        string = ''

        # Display the sign if required
        if sgn:
            if self.sign:
                string += '+ \n'
            else:
                string += '- \n'

        # Convert lists to a more readable string
        for i in range(len(self.llists)):
            for j in range(len(self.llists[i])):
                string += '[%d]' % self.llists[i][j]
            string += '\n'

        return string

    def disp_list(tab_list, sgn=False):
        # Displays each YT in a list
        for tableau in tab_list:
            print(YT.disp(tableau, sgn))

    def YoungD(self):
        # Returns the Young diagram corresponding to the tableau
        return YD([len(lis) for lis in self.llists])

    # ------------------------------------------------------------
    # Garnir algorithm
    def Garnir(self):
        # Permutation * YT is represented as a YT
        # This function returns a step closer to standard permutations
        # ! YT used for Garnir are not projection operators !

        # Takes a tableau with ordered rows

        # Find position of first entry greater than the one below
        pos = YT.check_standard(self)

        # If pos is 0, then tab is standard and we return False
        if not pos:
            return False

        # Otherwise we run the Garnir algorithm

        # Obtain A and B from Garnir algorithm and concatenate
        A = self.llists[pos[0]][pos[1]:]
        B = self.llists[pos[0] + 1][:pos[1] + 1]
        C = A + B

        # Permute elements in C
        Sn = list(symmetric(len(C)))[1:]  # Gives us non-identity permutations
        perm_C = []

        # Only add those with column ascent
        for perm in Sn:
            d = perm(C)
            if asc(d[:len(A)]) and asc(d[len(A):]):
                perm_C.append(d)

        # Print corresonding tableaux
        tableau_list = []
        for i in range(len(perm_C)):
            d = perm_C[i]
            Young = copy.deepcopy(self.llists)
            Young[pos[0]][pos[1]:] = d[:len(A)]
            Young[pos[0] + 1][:pos[1] + 1] = d[len(A):]

            # Sign for the tableaux we return will be negative unless len(C) = 2
            Young = YT(Young, False, not self.sign) # (len(C) == 2) ==
            tableau_list.append(YT.reorder_rows(Young))

        return tableau_list

    def Garnir_recur(self):
        # Returns full Garnir decomposition
        stand_list = []

        # First reorder rows:
        tab = YT.reorder_rows(self)

        non_stand_list = [tab]
        new_non_stand_list = []

        # Iterate Garnir
        while len(non_stand_list) > 0:
            for tableau in non_stand_list:
                if not YT.Garnir(tableau):
                    stand_list.append(tableau)

                else:
                    new_non_stand_list += YT.Garnir(tableau)

            non_stand_list = new_non_stand_list
            new_non_stand_list = []

        return stand_list

    # Garnir related functions
    def reorder_rows(self):
        # Orders rows in ascending order
        return YT([reorder_asc(lst) for lst in self.llists], False, self.sign)

    def check_standard(self, start=[0, 0]):
        # Given a YT with ascending rows:
        # Returns 0 if tableau is standard
        # Else returns position of first entry < the one below
        pos = 0
        lsts = self.transpose

        for i in range(start[0], len(lsts)):
            for j in range(start[1], len(lsts[i]) - 1):
                if lsts[i][j] > lsts[i][j + 1]:
                    pos = [j]
                    break
            if pos:
                pos.append(i)
                break

        return pos

    # ------------------------------------------------------
    # Generation of standard tableaux
    def gen_stand(YoungD, n0=0, check=True, pront=False):
        # Given a Young diagram, generates the standard tableaux
        # Adds n0 to all boxes at the end
        shape = copy.deepcopy(YoungD.part)
        stands = [[[1]]]

        # Add numbers 2 to n one at a time:
        for i in range(2, YoungD.n + 1):
            new_stands = []
            for k in range(len(stands)):
                tab = copy.deepcopy(stands[k])

                # Add i to rows which are both shorter than the row above
                # and shorter than the equivalent row of the Young diagram
                our_js = [j for j in range(len(tab))
                          if len(tab[j]) < len(tab[j - 1])
                          and len(tab[j]) < YoungD.part[j]
                          or j == 0 and len(tab[j]) < YoungD.part[j]]

                # Add a new row [i] if the number of rows is less than
                # the number of rows of the Young diagram
                for j in our_js:
                    new_tab = copy.deepcopy(tab)
                    new_tab[j].append(i)
                    new_stands.append(new_tab)
                if len(tab) < len(YoungD.part):
                    new_stands.append(tab + [[i]])

            stands = new_stands  # Update our tableaux

        # If n0, add n0 to all elements in list
        if n0:
            stands = [[[x + n0 for x in ls] for ls in stand] for stand in stands]

        # Convert lists to Young tableaux
        stands = [YT(tab) for tab in stands]

        if check:
            assert len(stands) == maths.factorial(YoungD.n) / YoungD.hook

        if pront:
            YT.disp_list(stands)

        return stands

    def gen_dict(stands):
        dict_1 = {YT.disp(stands[i]): 'e' + str(i) for i in range(len(stands))}

        return dict_1

    # ------------------------------------------------
    # Operators to permutations
    def op_to_perms(self, mode=False):
        # yields all the permutations of the Young operator given by the tableau
        m = 1
        sym = rep.lst_lst_to_perms(self.llists, m)
        asym = rep.lst_lst_to_perms(self.transpose, m)

        if not mode:
            for s1 in sym:
                for a1 in asym:
                    yield [mult_perms(s1, a1), int(a1.signature())]

    def lst_op_to_perms(ops):
        # Given [op1, ..., opn], returns the permutations of opn o ... o op1
        start = list(YT.op_to_perms(ops[0]))
        i = 1
        while i < len(ops):
            new = [[mult_perms(y[0], x[0]), x[1] * y[1]] for x in start for y in list(YT.op_to_perms(ops[i]))]
            start = new
            i += 1

        return start

    def permute(self, g):
        # Permutes numbers in a tableau according to g

        # Remember shape
        shape = [len(lst) for lst in self.llists]

        # Create single list out of young tableau
        single_list = [item - 1 for sublist in self.llists for item in sublist]

        h = pos_create_perm(single_list, range(self.n))

        # Act by h^(-1)gh and readd the 1
        single_list = [element + 1 for element in (h * g * h ** (-1))(single_list)]

        # Turn list into YT
        YT_lst = [[single_list[i + sum(shape[0:l])]
                   for i in range(shape[l])] for l in range(len(shape))]

        return YT(YT_lst, False, self.sign)

    # -------------------------------------------------------------------
    # Uniqueness
    def M(self, old_perm):
        # Given a young tableau and a permutation, returns m = -1, 0 or 1
        perm = (old_perm)**-1 # This is due to my convention
        try:
            lst = YT.order_sym(self)  # Returns which symmetriser numbers 1, ..., n are attached to
            lst = perm(lst)  # lst now contains which symmetriser each node is attached to after the permutation

            groups = copy.deepcopy(self.transpose)
            n_anti = len(self.transpose)  # Number of antisymmetrisers
            signature = 1

            # groups[i][j] will contain which symmetriser node j of the antisymmetriser i is connected to
            for i in range(n_anti):
                for j in range(len(self.transpose[i])):
                    groups[i][j] = lst[self.transpose[i][j] - 1]

                # Obtain the sign we get from reordering the antisymmetriser
                signature *= Permutation(np.array(groups[i]) - 1).signature()

            return signature

        except ValueError:
            # If two numbers are the same, this means 2 nodes of an antisymmetriser connect to the same symmetriser.
            # In this case, the result is zero.

            return 0

    def order_sym(self):
        # Returns a list containing the row of 1, ..., n
        llist = np.array([0] * self.n)
        for i in range(len(self.llists)):
            pos = np.array(self.llists[i]) - 1
            llist[pos] = i + 1

        return llist

    # --------------------------------------------------
    def pop_highest(self):
        # Returns the tableau with its greatest entry removed
        new = copy.deepcopy(self.llists)
        candidates = [lst[-1] for lst in new]

        if len(new[pos_highest(candidates)]) == 1:
            new.pop(pos_highest(candidates))

        else:
            new[pos_highest(candidates)].pop(-1)

        return YT(new, False, self.sign)


##
class rep:
    ''' Representation of S(n) furnished by a Young tableau

        Optional
        Extended
        Description

        Parameters
        ----------
        YoungT : class YT
            Young tableau from which the representation is to be generated
        gen_elements = False
            If true, generates all elements of S(n)
        check = False
            If true, carries out check
        check2 = False
            If true, carries out check

        Attributes
        ----------
        YoungT : class YT
            Young tableau from which the representation is generated
        YoungD : class YD
            Young diagram of the Young tableau
        stand_tabs : list of class YT
            List of the standard tableaux of shape YoungD
        dim : int
            Number of standard tableaux (dimension of the representation)
        n : int
            Number of boxes of the Young tableau
        perms : list of type Permutation
            Standard permutations corresponding to the standard tableaux
        g_dict : dict
            {permutation : matrix representation of that permutation}
        '''

    def __init__(self, YoungT, gen_elements = False, check = False, check2 = False):
        self.YoungT = YoungT
        self.YoungD = YT.YoungD(YoungT)
        self.stand_tabs = YT.gen_stand(self.YoungD)
        self.dim = len(self.stand_tabs)
        self.stand_dict = YT.gen_dict(self.stand_tabs)  # disp to e_i
        self.perms = rep.new_stand_perms(self)
        if gen_elements:
            self.g_dict = {perm: rep.rep_g(self, perm) for perm in list(symmetric(self.YoungD.n))}
        if not gen_elements:
            self.g_dict = {perm: None for perm in list(symmetric(self.YoungD.n))}
        if check:
            rep.check(self)
        if check2:
            rep.Check_garnir_2(self)

    # ----------------------------------------------------------------------------
    # Garnir functions
    def gen_vec(self, perm):
        # Expresses a Young tableau as a linear combination of standard tableaux
        permuted = YT.permute(self.YoungT, perm)

        vec = [0] * self.dim

        stands = YT.Garnir_recur(permuted)

        my_list = [self.stand_dict[YT.disp(tab)] for tab in stands]

        for i in range(len(my_list)):
            vec[int(my_list[i][1:])] += 2 * int(stands[i].sign) - 1

        return vec

    def lst_non_stand_to_stand(self, non_stand_list, stand='rectangles'):
        # Given a list of non-standard permutations
        # Returns as a sum of standard permutations, in the form of a vector
        # Option to instead act on a standard permutation of the YT of the rep
        vec = [0] * self.dim
        for x in non_stand_list:
            perm = x[0] * Permutation(self.YoungD.n - 1)  # Can shorten

            if not stand == 'rectangles':
                perm = mult_perms(perm, stand)

            sgn = x[1]
            vec = [sum(y) for y in zip(vec, [sgn * n for n in rep.gen_vec(self, perm)])]

        return vec

    # -------------------------------------------------------------
    # Conversion perms <-> tabs
    def new_stand_perms(self):
        # Expresses standard tableaux as permutations * rep_stand

        # Create single list out of young tableau
        official_stand = [item - 1 for sublist in self.YoungT.llists for item in sublist]

        permutations = []

        for stand in self.stand_tabs:
            single_lst_stand = [item - 1 for sublist in stand.llists for item in sublist]

            # Obtain the permutation p such that:
            # p(official_stand) = stand
            permutations.append(num_create_perm(official_stand, single_lst_stand))

        return permutations

    def stand_perms(self, check=False):
        # Expresses standard tableaux as permutations * self.YoungT
        # Returns the permutations

        permutations = []

        projector = copy.deepcopy(self.YoungT)

        for stand in [self.stand_tabs[0]]:
            perm_stand = Permutation(self.YoungD.n - 1)  # Identity permutation

            for col_n in range(len(projector.llists)):
                for row_n in range(len(projector.llists[col_n])):
                    p = projector.llists[col_n][row_n] - 1
                    s = stand.llists[col_n][row_n] - 1

                    if not perm_stand(p) == s:
                        perm_stand *= Permutation([[p, s], [self.YoungD.n - 1]])

            permutations.append(perm_stand)

        # Checks:
        if check:
            print(YT.disp(self.YoungT))
            for i in range(len(permutations)):
                perm = permutations[i]
                print(perm)
                print(YT.disp(YT.permute(self.YoungT, perm)))
                print(YT.disp(self.stand_tabs[i]))

        return permutations

    # -----------------------------
    # lsts to perms
    # these are all for the op_to_perms function
    def lst_lst_to_perms(lst_lst, N):
        # Takes a list of lists (intended: .llists for a YT object) and returns the symmetriser

        # Identity
        perms = [Permutation(N - 1)]

        for lst in lst_lst:
            # Obtain the symmetriser for each row
            lst_perms = rep.lst_to_perms(lst, N)

            # Multiply with the symmetrisers of the previous rows
            # Each number appears at most once across all rows, so this is
            # just concatenation of cycles
            perms = [old_perm * new_perm for old_perm in perms for new_perm in lst_perms]

        return perms

    def lst_to_perms(lst, N):
        # Takes a list (ex: a row of a YT) and returns a list of partitions into sets
        perm_lst = []
        for n, p in enumerate(set_partition(lst), 1):
            group_el = [[x - 1 for x in lst] for lst in sorted(p)]
            perm_lst.append(group_el)

        new_lst = []
        for perm in perm_lst:
            new_lst += rep.set_to_perms(perm, N)

        return new_lst

    def set_to_perms(part, N):
        # Takes a partition
        lst = [Permutation(N - 1)]
        for cycle in part:
            possibilities = rep.chosen_to_perms(cycle, N)
            lst = [old * poss for old in lst for poss in possibilities]

        return lst

    def chosen_to_perms(lst, N):
        # Takes a set (which represents a cycle), returns all cycles with same numbers and length
        # i.e. takes set (123) and returns permutations (123), (132)
        L = len(lst)

        if L <= 1:
            return [Permutation([lst, [N - 1]])]

        else:
            l0 = lst[0]
            sym_lst = list(symmetric(L - 1))
            perms = [Permutation([[l0] + sym(lst[1:]), [N - 1]]) for sym in sym_lst]

            return perms

    # ------------------------------------------------------
    # Garnir tests
    # For a list of permutations, checks their garnir decomposition
    # Better check than the previous Check_Garnir
    def Check_garnir_2(self):
        Sn = list(symmetric(self.YoungT.n))
        for perm in Sn:
            assert rep.check_perm_lst_action(self, [[perm, 1]])

        print('Each element correctly expressed as a sum of standard permutations')

    def check_perm_lst_action(self, act_lst, stand_vec = False):
        # Checks the Garnir decomposition of perm_lst acting on self.YoungT
        n = self.YoungT.n

        # Make a dictionary out of the elements of Sn
        dictionary = dict(zip(list(symmetric(n)), range(maths.factorial(n))))

        # Make the acting tableau and the standard tableau onto which it acts into permutations
        receive_lst = list(YT.op_to_perms(self.YoungT))

        # First vector: each permutation done individually
        vec1 = [0] * maths.factorial(n)
        for act in act_lst:
            for rec in receive_lst:
                res_perm = mult_perms(act[0], rec[0])
                vec1[dictionary[res_perm]] += act[1] * rec[1]

        # Second vector: first make the acting tableau into standard permutations
        vec2 = [0] * maths.factorial(n)
        if not stand_vec:
            stand_vec = rep.lst_non_stand_to_stand(self, act_lst)

        for i in range(len(self.perms)):
            stand = self.perms[i]
            for rec in receive_lst:
                res_perm = mult_perms(stand, rec[0])
                vec2[dictionary[res_perm]] += rec[1] * stand_vec[i]

        return np.array_equal(vec1, vec2)

    def check_tab_action(self, tab, stand_vec = False):
        # Checks the Garnir decomposition of an entire tableau
        rep.check_perm_lst_action(self, list(YT.op_to_perms(tab)), stand_vec)

    # -----------------------------------------------
    # Representation matrices
    def rep_g(self, g):
        # Returns the matrix representation of g
        mat = np.zeros([self.dim, self.dim]).astype(int)

        for i in range(len(self.stand_tabs)):
            stand_perm = self.perms[i]
            vec = rep.gen_vec(self, mult_perms(g, stand_perm))
            mat[i, :] = vec[:]

        return mat.transpose()

    def get_rep_g(self, g):
        # If the representation of g has been generated (and is in g_dict), returns the representation
        # Else generates the representation (adds it to g_dict) then returns the representation
        g *= Permutation(self.YoungT.n - 1)

        # Generate element if it hasn't been done
        if self.g_dict[g] is None:
            self.g_dict[g] = rep.rep_g(self, g)

        return self.g_dict[g]

    # ------------------------------------------------
    # Representation tests
    def check(self):
        for perm1 in list(symmetric(self.YoungD.n)):
            for perm2 in list(symmetric(self.YoungD.n)):
                assert np.array_equal(np.matmul(self.g_dict[perm1], self.g_dict[perm2]),
                                      self.g_dict[mult_perms(perm1, perm2)])
        print('The representation works !')

    # --------------------------------------------------
    # 3js
    def M_list(self, Young_tab, perm_list):
        # Given a list of permuted tableaux, gives the sum of their m = -1, 0, 1 values
        # of the correspondng standard permutations

        # First calculate the m value of each standard permutation
        m_stands = rep.M_stands(self, Young_tab)

        # Express the sum of permuted tableaux as a sum of standard tableaux
        vec = rep.lst_non_stand_to_stand(self, perm_list)

        # Multiply the number of each standard permutation with its m number
        m = 0
        for i in range(self.dim):
            m += vec[i] * m_stands[i]

        # m = Fraction(m, counter)

        return m

    def M_stands(self, projector, prnt=False):
        # Returns a list containing m(projector; standard permutation) for each standard tableau
        m_stands = [YT.M(projector, perm) for perm in self.perms]

        return m_stands

    def III_J(self, op1, op2, prnt=False):
        # Returns the 3j coefficient (op1 and op2 connected to self.YoungT)
        Y = copy.deepcopy(self.YoungT)

        # n-dependence
        n_dep, lam = YD.dimension(self.YoungD)

        # If either op1 or op2 has more columns or rows than Y,
        # then the 3j is 0

        if rep.op_too_big(op1, Y) or rep.op_too_big(op2, Y):
            return 0

        # M-value
        m1 = rep.M_list(self, Y, YT.op_to_perms(op1))
        m2 = rep.M_list(self, Y, YT.op_to_perms(op2))

        # There is possible redundancy here
        norm = YD.norm(YT.YoungD(op1)) * YD.norm(YT.YoungD(op2))

        III_J = m1 * m2 * n_dep * norm

        if prnt:
            print('m value for op1: ' + str(m1))
            print('m value for op2: ' + str(m2))

        return III_J

    # ----------------------------------------------------
    # Speeding up 3js
    # test_3j_rep3 = rep(YT([[1, 2, 5], [3, 4, 6]]))
    # ~7s
    # after adding op_too_big: ~3.5s
    def op_too_big(op, proj):

        return rep.L_too_big(op, proj) or rep.R_too_big(proj, op)

    def L_too_big(L, R):
        # Returns true if L*R = 0 on grounds of it not being possible
        # for R's symmetrisers to connect to different symmetrisers of L
        for i in range(len(L.transpose)):
            try:
                lst_L, lst_R = L.transpose[i], R.transpose[i]
                if len(lst_L) > len(lst_R):
                    return True
                if len(lst_L) < len(lst_R):
                    return False

            except IndexError:
                return True

        return False

    def R_too_big(L, R):
        # Returns true if L*R = 0 on grounds of it not being possible
        # for R's symmetrisers to connect to different symmetrisers of L
        for i in range(len(L.llists)):
            try:
                lst_L, lst_R = L.llists[i], R.llists[i]
                if len(lst_L) > len(lst_R):
                    return False
                if len(lst_L) < len(lst_R):
                    return True

            except IndexError:
                return False

        return True

    # ----------------------------------------------------
    # 3j checks
    def n_boxes(n, n0=0, disp=False):
        # Returns the n-box standard tableaux
        n_box_stands = []
        parts = partitions(n)
        for part in parts:
            for stand in YT.gen_stand(YD(part), n0):
                n_box_stands.append(stand)

        if disp:
            YT.disp_list(n_box_stands)

        return n_box_stands

    def check_III_J(self, pront=False):
        # Checks the sum rule

        Y = self.YoungT

        dim, lam = YD.dimension(YT.YoungD(Y))
        print('Dim Y is: ' + str(dim))
        print('Y has ' + str(Y.n) + ' boxes.')
        tot = 0

        for i in range(1, Y.n):
            X_list = rep.n_boxes(i)
            Z_list = rep.n_boxes(Y.n - i, i)

            for op1 in X_list:
                for op2 in Z_list:
                    J = rep.III_J(self, op1, op2)
                    tot += J
        if pront:
            print('The sum comes to: ' + str(tot))
            print('We expect: ' + str((Y.n - 1) * dim))
            print(tot == (Y.n - 1) * dim)
        assert tot == (Y.n - 1) * dim
        print('Representation passes the sum test !')

    def check_III_J_2(self):
        # Make tabX * tabY into a list of perms and add M values individually
        Y = self.YoungT
        for i in range(1, Y.n):
            X_list = rep.n_boxes(i)
            Z_list = rep.n_boxes(Y.n - i, i)

            for op1 in X_list:
                for op2 in Z_list:
                    # Method used in 3js
                    m1 = rep.M_list(self, Y, YT.op_to_perms(op1))
                    m2 = rep.M_list(self, Y, YT.op_to_perms(op2))

                    # Safer, slower method
                    m = 0
                    for x in YT.lst_op_to_perms([op1, op2]):
                        perm, sig = x[0], x[1]
                        m += sig * YT.M(Y, perm)

                    assert m1 * m2 == m

        print('Individually calculating each m gives the same result')

    # ---------------------------------------------------
    # 6js
    def Create_Mat(self, tab):
        # Create a matrix out of a tableau
        all_perms = YT.op_to_perms(tab)
        mat = sum([x[1] * rep.get_rep_g(self, x[0]) for x in all_perms])

        return mat

    def Matrix_VI_J(self, U, V, W, X, Z, check1 = False, check2 = False):
        # Returns the 6j, where the biggest operator is self.YoungT

        # n-dependence
        n_dep, lam = YD.dimension(self.YoungD)

        # Expresse Young tableaux as matrices in the matrix representation
        Z_Mat = rep.Create_Mat(self, Z)
        X_Mat = rep.Create_Mat(self, X)
        W_Mat = rep.Create_Mat(self, W)
        V_Mat = rep.Create_Mat(self, V)
        U_Mat = rep.Create_Mat(self, U)

        big_mat = reduce(np.dot, [Z_Mat, X_Mat, W_Mat, V_Mat, U_Mat])  # Multiply matrices

        # Create a vector of zeros with a 1 in the position of the identity permutation
        small_vec = [0]*len(self.perms)
        small_vec[self.perms.index(Permutation(self.YoungT.n - 1))] = 1

        # Act upon the vector representing the sandwiching projector
        # With the matrix corresponding to the joint effort of all the operators
        # Calculate m from the Garnir decomposition
        m = int(sum([np.product(y) for y in zip(big_mat.dot(small_vec), rep.M_stands(self, self.YoungT))]))

        # Optional checks, see individual functions for details
        if check1:
            assert m == rep.Slow_VI_J(self, U, V, W, X, Z)
            print('checked by multiplying out operators and calculating m for each individual permutation !')

        if check2:
            assert m == rep.Med_VI_J(self, U, V, W, X, Z)
            print('checked by multiplying operators first then applying Ganir !')

        norm = Fraction(1, np.product([YT.YoungD(tab).hook for tab in [U, V, W, X, Z]]))
        VI_J = norm * m * n_dep

        return VI_J

    # Checks
    def Sum_Test_VI_J(self):
        # n-dependence
        Y = self.YoungT
        n_dep, lam = YD.dimension(self.YoungD)
        coef = Fraction((self.YoungT.n - 1) * (self.YoungT.n - 2), 2)
        J = 0
        for i in range(2, Y.n):
            Z_list = rep.n_boxes(i)
            X_list = rep.n_boxes(Y.n - i, i)

            for j in range(1, i):
                V_list = rep.n_boxes(j)
                W_list = rep.n_boxes(i - j, j)
                U_list = rep.n_boxes(i - j, j + Y.n - i)

                J += sum([rep.Matrix_VI_J(self, U, V, W, X, Z) for U in U_list for V in V_list
                          for W in W_list for X in X_list for Z in Z_list])
        assert coef * n_dep == J
        print('Sum test passed :)))')

    def Slow_VI_J(self, U, V, W, X, Z):
        # Multiplies out tableaux into permutations
        # Calculates m individually for the permutations
        all_perms = YT.lst_op_to_perms([U, V, W, X, Z])
        m = sum([x[1] * YT.M(self.YoungT, x[0]) for x in all_perms])

        return m

    def Med_VI_J(self, U, V, W, X, Z, check = False):
        # Expresses Young tableaux as a big sum of permutations
        # Applies Garnir to all of these then calculates M
        all_perms = YT.lst_op_to_perms([U, V, W, X, Z])

        # Can check Garnir, although the same check should already
        # be applied preb=viously
        if check:
            rep.check_perm_lst_action(self, all_perms)

        return rep.M_list(self, self.YoungT, all_perms)

    def act_on_vec(self, vec, tab):
        # Computes the action of the YT tab on a linear combination of standard
        # permutations of the YT of the representation self
        # Returns a vector of standard permutation

        # Obtain the tableau as a sum of permutations:
        untouched_perms = list(YT.op_to_perms(tab))

        new_vec = [0] * len(vec)

        for i in range(len(vec)):
            new_vec = [sum(y) for y in zip(
                new_vec,
                [vec[i] * x for x in rep.lst_non_stand_to_stand(self, untouched_perms, self.perms[i])])]

        return new_vec

    # -------------------------------------------------------------
    # Hermitian matrices
    def herm(self, tab):
        if tab.n ==1:
            return rep.Create_Mat(self, tab)

        mat_lst = rep.gen_mat_lst(self, tab)
        op_lst = flatten(rep.nested_herm(tab.n))
        op_lst = [mat_lst[i] for i in op_lst]
        return reduce(np.dot, op_lst)

    def nested_herm(n, i = 0):
        if n == 2:
            return [i]

        else:
            return [rep.nested_herm(n-1, i+1), i, rep.nested_herm(n-1, i+1)]

    def gen_mat_lst(self, tab):
        mat_lst = []
        tab_copy = copy.deepcopy(tab)
        while tab.n >= 2:
            mat_lst.append(rep.Create_Mat(self, tab))
            tab = YT.pop_highest(tab)

        return mat_lst


## Class Young Diagram
class YD:
    ''' Young diagram generated from a partition of n

            Optional
            Extended
            Description

            Parameters
            ----------
            partition : list of int
                Partition from which the Young diagram is to be generated

            Attributes
            ----------
            part : list of int
                Partition from which the Young diagram was generated
            transpose : list of int
                Partition representing the transpose of the Young diagram
            n : int
                Number of boxes of the Young diagram
            hook : int
                Hook number of the Young diagram
            '''

    def __init__(self, partition):
        self.part = partition
        self.transpose = YD.transpose(self)
        self.n = sum(partition)
        self.hook = YD.hook(self)

    def dimension(self, solve=False):
        # Returns the dimension of the representation
        dimension = 1
        x = symbols('n')
        for i in range(len(self.part)):
            n = self.part[i]
            for j in range(n):
                dimension *= (x - i + j)

        dimension *= Fraction(1, YD.hook(self))

        if not solve:
            # Lambdify expression
            # Saves time if evaluated multiple times

            return dimension, sp.lambdify(x, dimension)

        else:

            return sp.lambdify(x, dimension)(solve)

    def norm(self):
        # Returns normalisation factor
        norm = 1

        return Fraction(norm, self.hook)

    def disp(self, transpose=False):
        # Displays the Young diagram
        if transpose:
            cat = self.transpose
        else:
            cat = self.part
        string = ''
        while len(cat) > 0:
            string += '[-]' * cat[0] + '\n'
            cat.pop(0)

        return string

    def hook(self, prnt=False):
        # Returns the Hook number of a Young diagram
        # Uses Young tableau class
        hooklist = []

        for n in self.part:
            l = list(range(1, n + 1))
            l.reverse()
            hooklist.append(l)
        tab = YT(hooklist)

        hklist = []
        for lis in tab.transpose:
            n = len(lis)
            l = list(range(0, n))
            l.reverse()
            hklist.append(l)
        tb = YT(hklist, True)

        hook = 1

        for i in range(len(hooklist)):
            for j in range(len(hooklist[i])):
                hook *= (tab.llists[i][j] + tb.llists[i][j])

        if prnt:
            print(hook)

        return hook

    def transpose(self):
        # Returns the transpose of a partition
        cat = copy.deepcopy(self.part)
        new_part = []
        counter = 1
        while len(cat) > 0:
            new_part.append(len(cat))
            cat = [x for x in cat if x != counter]  # Remove elements = counter
            counter += 1

        return new_part

    # ---------------------------------------------------
    # Representation tests
    def check_herm_orthog(self):
        stands = YT.gen_stand(self)
        mats = [rep.herm(rep(stand), stand) for stand in stands]
        n = len(mats)
        for i in range(n):
            for j in list(range(i)) + list(range(i+1, n)):
                mat1 = mats[i]
                mat2 = mats[j]
                assert np.array_equal(np.matmul(mat1, mat2), np.zeros((n, n)).astype(int))
        print('The hermitian matrices are orthogonal !')





##


##




## Testing representation, testing on Garnir by generating representation
test_rep = rep(YT([[1, 3, 5], [2, 4]]), True, True, True)


## Example 3j calculation
op1 = YT([[1,3], [2]])
op2 = YT([[4, 5]])
print(rep.III_J(test_rep, op1, op2))
rep.III_J(test_rep, op1, op2)



## Testing 3js with the sum rule
test_3j_rep3 = rep(YT([[1, 2, 5], [3, 4, 6], [7]]))
rep.check_III_J(test_3j_rep3)


## Different 3j test, we calculate m individually for each permutation
# Does not involve Garnir
rep.check_III_J_2(test_3j_rep3)


## Testing VI_J
test_U = YT([[2, 3], [4]])
test_V = YT([[1]])
test_W = YT([[2]])
test_X = YT([[3], [4]])
test_Z = YT([[1], [2]])
test_Y = YT([[1, 3], [2], [4]])
test_6j_rep = rep(test_Y)
#YT.disp_list([test_U, test_V, test_W, test_X, test_Z, test_Y])
print(rep.Matrix_VI_J(test_6j_rep, test_U, test_V, test_W, test_X, test_Z, True, True))


## VI_J sum test
test_Y = YT([[1, 3], [2], [4], [5]])
test_6j_rep = rep(test_Y)
rep.Sum_Test_VI_J(test_6j_rep)
##

YD.check_herm_orthog(YD([3, 2]))
##

