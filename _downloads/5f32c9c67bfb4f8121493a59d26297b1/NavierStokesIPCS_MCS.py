from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
realcompile=True

__all__ = ["NavierStokes"]

# method from:
# https://epubs.siam.org/doi/pdf/10.1137/19M1248960

# Incremental Pressure-Correction Scheme


ngsglobals.symbolic_integrator_uses_diff = True

class NavierStokes:
    
    def __init__(self, mesh, nu, inflow, outflow, wall, uin, timestep, meshvelocity=CF((0,0,0)), rotor="rotor",
                 Omega=CF((0,0,0)), friction=None, order=2, substeps=None, verbose=0,
                 resistance_outflow=None):

        self.nu = nu
        self.timestep = timestep
        self.uin = CF(uin) # -CF(meshvelocity)
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall
        self.friction = friction
        self.verbose = verbose
        self.resistance_outflow = resistance_outflow

        # self.meshvelocity = GridFunction(HCurl(mesh, order=5)) # precise on curved mesh
        # self.meshvelocity.Set(meshvelocity, dual=True)
        # self.Omega = 0.5*curl(self.meshvelocity)
        self.meshvelocity = meshvelocity

        self.boundaryvalues=mesh.BoundaryCF( { self.inflow:self.uin-self.meshvelocity ,
                                                 self.wall+'|'+self.outflow:-self.meshvelocity }, default=CF((0,0,0)))

        
        self.useRT = False
        V = HDiv(mesh, order=order, dirichlet=inflow+"|"+wall+"|"+rotor, RT=self.useRT, highest_order_dc=True)
        Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet=inflow+"|"+wall+"|"+rotor+"|"+outflow)
        Sigma = Discontinuous(HCurlDiv(mesh, order = order-1, orderinner=order))
        S = MatrixValued(L2(mesh, order=order-1), skewsymmetric=True)
        self.Q = L2(mesh, order=order-1+(1 if self.useRT else 0))

        
        self.V = V
        Sigma = Compress(PrivateSpace(Sigma))
        S = Compress(PrivateSpace(S))
        
        self.X = V*Vhat*Sigma*S
        if self.verbose >= 1:
            print ("ndof X =", self.X.ndof, " = ", V.ndof, "+", Vhat.ndof) # , Sigma.ndof, S.ndof)
        
        for i in range(self.X.ndof):
            if self.X.CouplingType(i) == COUPLING_TYPE.WIREBASKET_DOF:
                self.X.SetCouplingType(i, COUPLING_TYPE.INTERFACE_DOF)
        
        u, uhat, sigma, W  = self.X.TrialFunction()
        v, vhat, tau, R  = self.X.TestFunction()

        dS = dx(element_boundary=True, bonus_intorder=2)
        n = specialcf.normal(mesh.dim)
        def tang(u): return u-(u*n)*n
        
        # stokesA = -0.5/nu * InnerProduct(sigma,tau) * dx + \
        #   (div(sigma)*v+div(tau)*u) * dx + \
        #   (InnerProduct(W,tau) + InnerProduct(R,sigma)) * dx + \
        #   -(((sigma*n)*n) * (v*n) + ((tau*n)*n )* (u*n)) * dS + \
        #   (-(sigma*n)*tang(vhat) - (tau*n)*tang(uhat)) * dS
        stokesA = -0.5*nu * InnerProduct(sigma,tau) * dx + \
            nu*(div(sigma)*v+div(tau)*u) * dx + \
            nu*(InnerProduct(W,tau) + InnerProduct(R,sigma)) * dx + \
            -nu*(((sigma*n)*n) * (v*n) + ((tau*n)*n )* (u*n)) * dS + \
            nu*(-(sigma*n)*tang(vhat) - (tau*n)*tang(uhat)) * dS

        if friction:
            stokesA += friction*u*v*dx
        
        
        self.astokes = BilinearForm (self.X, eliminate_hidden = True, condense=True, store_inner=True)
        self.astokes += stokesA
        self.astokes += 1e8*nu*div(u)*div(v) * dx

        self.a = BilinearForm (self.X, eliminate_hidden = True)
        self.a += stokesA

        self.div = BilinearForm(div(u)*self.Q.TestFunction()*dx).Assemble()


        
        self.gfu = GridFunction(self.X)
        self.f = LinearForm(self.X)

        self.mstar = BilinearForm(self.X, eliminate_hidden = True, condense=True)
        self.mstar += u*v * dx + timestep * stokesA

        self.premstar = preconditioners.BDDC(self.mstar)
        self.mstar.Assemble()
        
        self.invmstar1 = solvers.CGSolver(self.mstar.mat, pre=self.premstar, atol=1e-7, printrates=False, maxiter=1000)
        ext = IdentityMatrix()+self.mstar.harmonic_extension
        extT = IdentityMatrix()+self.mstar.harmonic_extension_trans
        self.invmstar = ext @ self.invmstar1 @ extT + self.mstar.inner_solve
        
        # the convective term

        if substeps:
            class ConvClass(BaseMatrix):
                def __init__ (self, X, timestep, substeps,  meshvelocity, uin):
                    super(ConvClass, self).__init__()
                    self.X = X
                    self.timestep = timestep
                    self.substeps = substeps
                    self.meshvelocity = meshvelocity
                    self.uin = uin

                    self.rest = X.Restriction(0)
                    
                    self.VL2 = VectorL2(mesh, order=order, piola=True, dgjumps=True, order_policy=ORDER_POLICY.CONSTANT)
                    self.wind = GridFunction(self.VL2)  # X.components[0])
                    ul2,vl2 = self.VL2.TnT()
                    utot = ul2+self.meshvelocity
                    windtot = self.wind  # +self.meshvelocity

                    self.conv_l2 = BilinearForm(self.VL2, nonassemble=True) # False, nonlinear_matrix_free_bdb=True)
                    self.conv_l2 += InnerProduct(Grad(vl2)*windtot, utot).Compile(realcompile=realcompile, wait=True) * dx
                    self.conv_l2 += Cross(Omega, utot)*vl2*dx
                    self.conv_l2 += (-IfPos(windtot * n, windtot*n*utot*vl2, \
                                    windtot*n*(ul2.Other(bnd=mesh.BoundaryCF({inflow:self.uin-meshvelocity, wall:-meshvelocity, rotor:CF((0,0,0))}, default=0.5*ul2))+meshvelocity)*vl2)) \
                        .Compile(realcompile=realcompile, wait=True) * dS

                    self.convertl2 = X.components[0].ConvertL2Operator(self.VL2) @ self.X.Restriction(0)
                    self.mass = self.VL2.Mass(1)
                    self.massinv = self.VL2.Mass(1).Inverse()
                    self.mpts = mesh.MapToAllElements(IntegrationRule(ET.TET, 5), VOL)

                    
                def Mult (self, x, y):
                    self.wind.vec.data = (self.convertl2*x).Evaluate()                    
                    xl2 = self.wind.vec


                    # element-wise strength of convection
                    mesh = self.VL2.mesh
                    abswind = Norm(Inv(specialcf.JacobianMatrix(3,3))*self.wind) (self.mpts)
                    abswind = np.amax(abswind.reshape((mesh.GetNE(VOL),-1)), axis=1)
                    
                    fastelements = BitArray(self.timestep*abswind > 0.3)
                    # print ("fast elements =", fastelements.NumSet(), "/", mesh.GetNE(VOL))
                    # print ("maxfast =", self.timestep*max(abswind))
                    fastdofs = BitArray(self.VL2.ndof)
                    fastdofs[:] = False
                    for el in self.VL2.Elements(VOL):
                        for d in el.dofs:
                            fastdofs[d]=True
                    projfast = Projector(fastdofs, True)

                    yl2 = xl2.CreateVector()
                    xnew = xl2.CreateVector()                    
                    Bx = xl2.CreateVector()

                    yl2[:] = 0
                    self.conv_l2.Apply(xl2, Bx)

                    yl2.data = Bx
                    xnew.data = xl2 + self.timestep * self.massinv*Bx

                    # for igs in self.conv_l2.integrators:
                    #    igs.SetDefinedOnElements(fastelements)
                    
                    # iterate to solve  (I-B)*xnew = x
                    for j in range(substeps):
                        self.conv_l2.Apply(xnew, Bx)
                        # xnew.data += 1/substeps * projfast*(xl2 - xnew + self.timestep*self.massinv*Bx).Evaluate()
                        xnew.data += 1/substeps * (xl2 - xnew + self.timestep*self.massinv*Bx).Evaluate()
                        yl2 *= (1-1/substeps)
                        yl2.data += 1/substeps * Bx

                    # for igs in self.conv_l2.integrators:
                    #     igs.SetDefinedOnElements(None)




                    
                    # gfdelta = GridFunction(self.VL2)
                    # gfdelta.vec.data = xnew-xl2
                    # import pickle
                    # outfile = open("incconvection.pkl", "wb")
                    # pickle.dump([gfdelta], outfile)
                    # print ("pickle convection term")

                    # try with GMRes
                    # self.conv_l2.Apply(xl2, Bx)
                    # xnew.data = xl2 + self.timestep * self.massinv*Bx
                    # iterate to solve  (I-B)*xnew = x
                    # mat = IdentityMatrix() - self.timestep*self.massinv@self.conv_l2.mat
                    # for j in range(substeps):
                    #    xnew.data += 1/substeps * (xl2 - mat * xnew).Evaluate()

                    # solvers.GMRes(A=mat, b=xl2, x=xnew, pre=IdentityMatrix(len(xl2)), printrates=True)
                    yl2 = 1/self.timestep*(self.mass*(xnew-xl2)).Evaluate()
                    
                    y.data = self.convertl2.T * yl2
                def Shape (self):
                    return self.conv_operator.shape


                
            # put old and new value into one space
            class ConvClass2(BaseMatrix):
                def __init__ (self, X, timestep, substeps,  meshvelocity, uin):
                    super(ConvClass2, self).__init__()
                    self.X = X
                    self.timestep = timestep
                    self.substeps = substeps
                    self.meshvelocity = meshvelocity
                    self.uin = uin

                    self.rest = X.Restriction(0)
                    
                    self.VL2 = VectorL2(mesh, order=order, piola=True, dgjumps=True)
                    self.VL22 = self.VL2*self.VL2
                    # self.wind = GridFunction(self.VL2)  # X.components[0])
                    (wind,ul2),(dwind,vl2) = self.VL22.TnT()
                    utot = ul2+self.meshvelocity
                    windtot = wind+self.meshvelocity

                    self.conv_l2_vol = BilinearForm(self.VL22, nonassemble=False, nonlinear_matrix_free_bdb=True)
                    self.conv_l2_vol += InnerProduct(Grad(vl2)*windtot, utot).Compile(realcompile=realcompile, wait=True) * dx
                    self.conv_l2_vol.Assemble()
                    
                    self.conv_l2 = BilinearForm(self.VL22, nonassemble=True) # False, nonlinear_matrix_free_bdb=True)
                    # self.conv_l2 += InnerProduct(Grad(vl2)*windtot, utot).Compile(realcompile=realcompile, wait=True) * dx
                    self.conv_l2 += (-IfPos(windtot * n, windtot*n*utot*vl2, \
                                    windtot*n*(ul2.Other(bnd=mesh.BoundaryCF({inflow:self.uin-meshvelocity, wall:-meshvelocity, rotor:CF((0,0,0))}, default=0.5*ul2))+meshvelocity)*vl2)) \
                        .Compile(realcompile=realcompile, wait=True) * dS

                    self.convertl2 = X.components[0].ConvertL2Operator(self.VL2) @ self.X.Restriction(0)
                    self.mass = self.VL2.Mass(1)
                    self.massinv = self.VL2.Mass(1).Inverse()
            
                def Mult (self, x, y):
                    xold1 = (self.convertl2*x).Evaluate()
                    nd1 = len(xold1)
                    xl2 = BaseVector(2*nd1)
                    xl2[0:len(xold1)] = xold1
                    xl2[len(xold1):] = xold1
                    
                    yl2 = xl2.CreateVector()
                    xnew = xl2.CreateVector()                    
                    Bx = xl2.CreateVector()

                    yl2[:] = 0
                    self.conv_l2.Apply(xl2, Bx)
                    Bx.data += self.conv_l2_vol.mat * xl2
                    # Bx.data += self.conv_l2_vol.mat * xl2
                    
                    yl2.data = Bx
                    xnew.data = xl2
                    xnew[nd1:] += self.timestep * self.massinv*Bx[nd1:]
                    
                    # iterate to solve  (I-B)*xnew = x
                    for j in range(substeps):
                        self.conv_l2.Apply(xnew, Bx)
                        Bx.data += self.conv_l2_vol.mat * xnew                        
                        xnew[nd1:] += 1/substeps * (xl2[nd1:] - xnew[nd1:] + self.timestep*self.massinv*Bx[nd1:]).Evaluate()
                        yl2 *= (1-1/substeps)
                        yl2.data += 1/substeps * Bx
                    
                    y.data = self.convertl2.T * yl2[nd1:]
                def Shape (self):
                    return self.conv_operator.shape

                
            self.conv_operator = ConvClass(self.X, self.timestep, substeps, self.meshvelocity, self.uin)
            

        else:
            if False:
                u,v = V.TnT()
                self.conv = BilinearForm(V, nonassemble=True)
                utot = u+meshvelocity
                self.conv += InnerProduct(Grad(v)*utot, utot).Compile(realcompile=realcompile, wait=True) * dx
                self.conv += (-IfPos(utot * n, utot*n*utot*v, \
                                     utot*n*(u.Other(bnd=mesh.BoundaryCF({inflow:uin-meshvelocity, wall:-meshvelocity, rotor:CF((0,0,0))}, default=0.5*u))+meshvelocity)*v)) \
                    .Compile(realcompile=realcompile, wait=True) * dS
                if self.resistance_outflow is not None:
                    normal = specialcf.normal(mesh.Materials(".*"))
                    self.conv += -self.resistance_outflow * (u.Trace() * n) * (u.Trace() * n) * (v.Trace() * n) * ds(outflow)
                    # OmegaXr = Cross(self.Omega, CF((x,y,z)))
                    # self.conv += Cross(self.Omega, 2*u + OmegaXr)*v*dx
                    # self.conv += -1*OmegaXr*OmegaXr * v*n * dS
                    # self.conv += 0.5*OmegaXr*(2*u+OmegaXr) * v*n * dS
                    
                rest = self.X.Restriction(0)
                self.conv_operator = rest.T @ self.conv.mat @ rest
            else:
                VL2 = VectorL2(mesh, order=order, piola=True)
                ul2,vl2 = VL2.TnT()
                utot = ul2+self.meshvelocity
                self.conv_l2 = BilinearForm(VL2, nonassemble=True)
                self.conv_l2 += InnerProduct(Grad(vl2)*utot, utot).Compile(realcompile=realcompile, wait=True) * dx(bonus_intorder=2)
                self.conv_l2 += (-IfPos(utot * n, utot*n*utot*vl2, \
                                        utot*n*(ul2.Other(bnd=mesh.BoundaryCF({inflow:self.uin-meshvelocity, wall:-meshvelocity, rotor:CF((0,0,0))}, default=0.5*ul2))+meshvelocity)*vl2)) \
                    .Compile(realcompile=realcompile, wait=True) * dS
                # OmegaXr = Cross(self.Omega, CF((x,y,z)))
                # self.conv_l2 += Cross(self.Omega, 2*ul2+OmegaXr)*vl2*dx
                # self.conv_l2 += -1*OmegaXr*OmegaXr * vl2*n * dS   # or factor 1/2 ? 
                self.convertl2 = V.ConvertL2Operator(VL2) @ self.X.Restriction(0)
                self.conv_operator = self.convertl2.T @ self.conv_l2.mat @ self.convertl2

            

        # setup problem for pressure projection (hybrid mixed)
        self.V2 = Discontinuous(self.V)
        self.gfp = GridFunction(self.Q)
        self.Qhat = FacetFESpace(mesh, order=order, dirichlet=outflow)        
        self.Xproj = self.V2*self.Q*self.Qhat
        (u,p,phat),(v,q,qhat) = self.Xproj.TnT()
        aproj = BilinearForm(self.Xproj, condense=True)
        aproj += (-u*v+ div(u)*q + div(v)*p) * dx + (u*n*qhat+v*n*phat) * dS


        
        if False:  
            cproj = preconditioners.BDDC(aproj, coarsetype="h1amg", coarseflags= { "maxcoarse" : 10000, "verbose" : self.verbose })
            aproj.Assemble()

        if True:
            # Auxiliary space preconditioner
            fesh1 = H1(mesh, order=1, dirichlet="outlet")
            uh1,vh1 = fesh1.TnT()
            conv = self.Xproj.embeddings[2]@ConvertOperator(fesh1, self.Qhat, geom_free=True)
            ah1 = BilinearForm(grad(uh1)*grad(vh1)*dx)
            preh1 = preconditioners.H1AMG (ah1, maxcoarse=1000, verbose=self.verbose)
            ah1.Assemble()
            aproj.Assemble()
            prefacet = preconditioners.Local(aproj, GS=False, blocktype=["facet"]).mat
            cproj = conv @ preh1 @ conv.T + prefacet

            
            

        # self.invproj1 = aproj.mat.Inverse(self.Xproj.FreeDofs(aproj.condense), inverse="sparsecholesky")
        self.invproj1 = solvers.CGSolver(aproj.mat, pre=cproj, printrates=False, tol=1e-8, maxiter=1000)
        ext = IdentityMatrix()+aproj.harmonic_extension
        self.invproj = ext @ self.invproj1 @ ext.T + aproj.inner_solve

        normal = specialcf.normal(mesh.Materials(".*"))        
        # self.bproj = BilinearForm(div(self.V.TrialFunction())*q*dx, geom_free=True).Assemble()
        self.bproj = BilinearForm(div(self.V.TrialFunction())*q*dx + self.V.TrialFunction()*normal*qhat*dx(element_boundary=True), geom_free=True).Assemble()

        self.rhsproj = LinearForm (self.boundaryvalues*normal*qhat.Trace()*ds).Assemble()

        # self.rhspropeller = LinearForm (self.meshvelocity*normal*qhat.Trace()*ds("ventilator")).Assemble()        
        # print ("norm rhspropeller = " , Norm(self.rhspropeller.vec))

        # mapping of discontinuous to continuous H(div)
        ind = self.V.ndof * [0]
        for el in mesh.Elements(VOL):
            dofs1 = self.V.GetDofNrs(el)
            dofs2 = self.V2.GetDofNrs(el)
            for d1,d2 in zip(dofs1,dofs2):
                ind[d1] = d2
        self.mapV = PermutationMatrix(self.Xproj.ndof, ind)
        self.a.Assemble()
        self.f.Assemble()
        
                
    @property
    def velocity(self):
        return self.gfu.components[0]
    @property
    def pressure(self):
        return self.gfp
        # return 1e6/self.nu*div(self.gfu.components[0])

    def Save(self, filename):
        self.gfu.vec.FV().NumPy().tofile(filename)

    def Load(self, filename):
        import numpy as np
        self.gfu.vec.FV().NumPy()[:] = np.fromfile(filename)
        self.gfp.Set(-1e8*self.nu*div(self.gfu.components[0]))
        
    def SolveInitial(self, direct=False):

        self.gfu.components[0].Set (self.boundaryvalues, definedon=self.X.mesh.Boundaries(self.inflow+"|"+self.wall))
        self.gfu.components[1].Set (self.boundaryvalues, definedon=self.X.mesh.Boundaries(self.inflow+"|"+self.wall+"|"+self.outflow))

        self.Project(self.gfu.components[0].vec, True)
        self.astokes.Assemble()
        if direct:
            inv = self.astokes.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")
        else:
            prestokes = preconditioners.Local(self.astokes, block=True, GS=True, blocktype=["facet", "vertexpatch:hdivlo"])
            inv = solvers.CGSolver(self.astokes.mat, pre=prestokes, printrates=self.verbose>=2, maxiter=1000, tol=1e-6)

        if (self.astokes.condense):
            ext = IdentityMatrix()+self.astokes.harmonic_extension
            extT = IdentityMatrix()+self.astokes.harmonic_extension_trans
            fullinv = ext @ inv @ extT + self.astokes.inner_solve

            extm = IdentityMatrix()-self.astokes.harmonic_extension
            extmT = IdentityMatrix()-self.astokes.harmonic_extension_trans
            fullastokes = extmT @ (self.astokes.mat+self.astokes.inner_matrix) @ extm 
        else:
            fullinv = inv
            fullastokes = self.astokes.mat
        
        rhs = (fullastokes * self.gfu.vec + self.f.vec).Evaluate()
        self.gfu.vec.data -= fullinv * rhs
        self.gfp.Set(-1e8*self.nu*div(self.gfu.components[0]))
                
    def AddForce(self, force):
        force = CF(force)
        v, vhat, tau, R  = self.X.TestFunction()        
        self.f += -force*v*dx
        
    def DoTimeStep(self):
        
        self.temp = self.a.mat.CreateColVector()
        self.temp2 = self.a.mat.CreateRowVector()
        self.f.Assemble()
        
        self.temp.data = self.conv_operator * self.gfu.vec
        self.temp.data += self.f.vec
        self.temp.data += -self.a.mat * self.gfu.vec
        
        self.temp.data += self.div.mat.T * self.gfp.vec
        
        self.temp2.data = self.invmstar*self.temp
        
        # self.ComputePressure (self.temp)
        self.Project(self.temp2, False)
        self.gfu.vec.data += self.timestep * self.temp2.data

    def Project(self,vel,usebndvals):
        emb = self.X.Embedding(0)
        rest = self.X.Restriction(0)
        if usebndvals:
            projsol = (self.invproj * (self.bproj.mat @ rest * vel - self.rhsproj.vec)).Evaluate()
        else:
            projsol = (self.invproj @ self.bproj.mat @ rest * vel).Evaluate()
            # projsol = (self.invproj * (self.bproj.mat @ rest * vel - self.rhspropeller.vec)).Evaluate()
        vel.data -= emb @ self.mapV * projsol
        # self.gfp.vec.data = -self.Xproj.Restriction(1)*projsol

        # self.gfp.vec.data *= (1-self.timestep)
        # self.gfp.vec.data -= self.timestep*self.Xproj.Restriction(1)*projsol
        self.gfp.vec.data -= self.Xproj.Restriction(1)*projsol
                

    def ComputePressure (self,rhsstep1):
        emb = self.X.Embedding(0)
        rest = self.X.Restriction(0)
        
        projsol = (self.invproj @ self.mapV.T @ emb.T) * rhsstep1
        # projsol = self.invproj * (self.mapV.T @ emb.T * rhsstep1 - self.rhspropeller.vec)
        self.gfp.vec.data = -self.Xproj.Restriction(1)*projsol
