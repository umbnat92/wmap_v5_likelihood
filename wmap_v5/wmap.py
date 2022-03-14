"""
.. module:: wmap_v5
:Author: Umberto Natale

"""
import os
from typing import Optional

import numpy as np
from scipy import linalg
from astropy.io import fits
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

import sys


class WMAP(InstallableLikelihood):
    install_options = {
        "download_url": "https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/likelihood/wmap_likelihood_full_v5.tar.gz",
        "data_path": "WMAP_v5",
    }

    data_folder: Optional[str] = "WMAP_v5/wmap_likelihood_v5/data/"
    beam_mode: Optional[str]
    ptsrc_mode: Optional[str]

    options: dict
    
    gibbs_sigma_filename: Optional[str] = "lowlT/gibbs/sigmaEllsHkeChu_test16_ilc_9yr_5deg_r5_2uK_corrected_kq85y9_June_r5_all.fits"
    gibbs_first_iteration: Optional[int] = 10
    gibbs_last_iteration: Optional[int] = 15000
    gibbs_skip: Optional[int] = 2
    gibbs_ell_max: Optional[int] = 32

    lowl_max: Optional[int] = 32

    ttmax: Optional[int] = 1200
    ttmin: Optional[int] = 2
    temax: Optional[int] = 800
    temin: Optional[int] = 2
    

    def initialize(self):
        self.ires = 3
        self.nsidemax = 2**self.ires
        self.nlmax = 3*self.nsidemax-1
        self.npix = 12 * self.nsidemax**2

        if (not getattr(self, "path", None)) and (not getattr(self, "packages_path", None)):
            raise LoggedError(
                self.log,
                f"No path given to WMAP data. Set the likelihood property 'path' or "
                "the common property '{_packages_path}'.",
            )
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                f"The 'data_folder' directory does not exist. Check the given path [{self.data_folder}].",
            )

        if self.options['use_lowl_pol']:
            self.initialize_lowl_pol_data()
            self.te_hi_l_start = 24
        else:
            self.te_hi_l_start = self.temin

        if self.options['use_lowl_TT']:
            self.tt_hi_l_start = self.lowl_max + 1
            if self.options['use_gibbs']:
                self.initialize_gibbs_for_tt()
        else:
            self.tt_hi_l_start = self.ttmin
            
        self.get_diag_terms()
        self.get_offdiag_terms()

        if self.options['use_lowTBEB'] and self.options['use_highl_TB']:
            self.use_cl = ['tt','te','ee','bb','tb','eb']
        else:
            self.use_cl = ['tt','te','ee','bb']

        
        self.log.info("Initialization done!")


    def loglike(self,cltt,clte,clee,clbb,cleb=None,cltb=None,**params_values):
        logp = 0.
        if self.options['use_lowl_TT']:
            logp += self.lowl_TT_likelihood(cltt)

        if self.options['use_lowl_pol']:
            logp += self.lowl_TEEEBB_likelihood(cltt,clte,clee,clbb)

        if self.options['use_TT'] or self.options['use_TE']:
            self.TT_TE_covariance(cltt,clte)

        if self.options['use_TT']:
            logp += self.TTTT_likelihood(cltt)

        if self.options['use_TE']:
            logp += self.TETE_likelihood(clte)

            
        return logp


    def lowl_TT_likelihood(self,cltt):
        if self.options['use_gibbs']:
            cl_gibbs = np.zeros(self.gibbs_ell_max)
            if self.gibbs_ell_max > self.lowl_max:
                cl_gibbs[self.lowl_max:self.gibbs_ell_max]  =  self.cl_tt_fiducial[self.lowl_max:self.gibbs_ell_max]
            cl_gibbs[self.ttmin:self.lowl_max] = cltt[self.ttmin:self.lowl_max]

            #compute_br_estimator() returns the natural logarithm of the
            #likelihood(plus some arbitrary constant).  The code expects
            #the negative of this quantity(= chisquared/2)
            logp = - self.compute_br_estimator(cl_gibbs[self.ttmin:self.gibbs_ell_max]) 
            return logp
        else:
            raise ValueError("At the moment the direct evaluation of likelihood in pixel space"
                "using a resolution 4 temperature map is not implemented. Set 'use_gibbs' to True.")


    def lowl_TEEEBB_likelihood(self,cltt,clte,clee,clbb):
        nlmax = 23
        Omega_pix = np.pi / (3.*8.**2.)
        ell = np.arange(2, nlmax+1)

        cltt[ell] /= ell*(ell+1) * (2.*np.pi) * 1e-6 * self.wl[ell]** 2.
        clte[ell] /= ell*(ell+1) * (2.*np.pi) * 1e-6 * self.wl[ell]** 2.
        clee[ell] /= ell*(ell+1) * (2.*np.pi) * 1e-6 * self.wl[ell]** 2.


        xxx = np.zeros(2*nlmax-2)
        xxx[ell-2] = clee[ell]-clte[ell]**2./cltt[ell]
        xxx[ell[-1]-1:] = clbb[ell]

        yyy = linalg.blas.dgemv(alpha=1.,a=self.ninvplninv,x=xxx)

        Dp = np.empty_like(self.Dp0)

        k = 0
        for jp in range(2*self.mp):
            for ip in range(jp):
                Dp[ip,jp] = self.Dp0[ip,jp] + yyy[k]
                k += 1

        Dp = linalg.lapack.dpotrf(Dp)[0]
        print(Dp)
        Dp = linalg.lapack.dpotri(Dp)[0]
        print(Dp)

        p_r3 = self.predict_QU_res3(cltt,clte,ell)

        m_r3 = np.zeros_like(p_r3,dtype=complex)
        m_r3 = self.w_r3 - p_r3
        zzz = m_r3
        m_r3 = linalg.lapack.dpotrs(Dp,m_r3)[0]

        print(m_r3)
        print(zzz)

        return m_r3 @ zzz


    def TTTT_likelihood(self,cltt):
        logp = 0.
        z = np.log(self.cltt_dat[:self.ttmax+1]+self.ntt[:self.ttmax+1])
        zbar = np.log(cltt[self.ttmin:self.ttmax+1]+self.ntt[:self.ttmax+1])
        for l2 in range(self.ttmax-self.ttmin):
            for l1 in range(l2):
                if l1 == l2:
                    if l2 <= self.temax:
                        fisher = self.tete[l1] / (self.tttt[l1]*self.tete[l1]-self.ttte[l1]**2)
                    else:
                        fisher = 1. / self.tttt[l1]
                else:
                    fisher = - self.R_off_tttt[l1,l2] / np.sqrt(self.tttt[l1]*self.tttt[l2])\
                             + self.epsilon[l1,l2] / (self.tttt[l1]*self.tttt[l2])
                off_log_curv = (cltt[l1+self.ttmin]+self.ntt[l1]) * fisher \
                                        * (cltt[l2+self.ttmin]+self.ntt[l2])

                like = 2./3. * (z[l1]-zbar[l1]) * off_log_curv * (z[l2]-zbar[l2]) \
                      +1./3. * (cltt[l1]-self.cltt_dat[l1]) * fisher * (cltt[l2]-self.cltt_dat[l2])
                if (l1 >= self.tt_hi_l_start) or (l2 >= self.tt_hi_l_start):
                    if l1 == l2:
                        logp += like
                    else:
                        logp += 2*like
        return logp/2.


    def TETE_likelihood(self,clte):
        logp = 0.
        for l2 in range(self.temax-self.temin):
            for l1 in range(l2):
                if l1 == l2:
                    if l2 <= self.temax:
                        fisher = self.tttt[l1] / (self.tttt[l1]*self.tete[l1]-self.ttte[l1]**2)
                    else:
                        fisher = 1. / self.tete[l1]
                else:
                    fisher = - self.R_off_tete[l1,l2] / np.sqrt(self.tete[l1]*self.tete[l2])\
                             + self.epsilon[l1,l2] / (self.tttt[l1]*self.tttt[l2])
                
                like = (clte[l1]-self.clte_dat[l1]) * fisher[l1,l2] * (clte[l2]-self.clte_dat[l2])

                if (l1 >= self.te_hi_l_start) or (l2 >= self.te_hi_l_start):
                    if l1 == l2:
                        logp += like
                    else:
                        logp += 2*like
        return logp/2.


    def predict_QU_res3(self,cltt,clte,ell):
        p_r3 = np.zeros(2*self.mp,dtype=complex)
        for ip in range(self.mp):
            i = 3
            for l in range(2,23+1):
                p_r3[ip] += clte[l]/cltt[l] * self.wl[l] * self.alm_tt[l,0] * self.NinvY[self.goodpix[ip], i]
                p_r3[ip+self.mp] += clte[l]/cltt[l] * self.wl[l] * self.alm_tt[l,0] * self.NinvY[self.goodpix[ip]+self.npix, i]
                i += 1
            for m in range(1,l+1):
                p_r3[ip] += clte[l]/cltt[l] * self.wl[l] * (self.alm_tt[l,m]*self.NinvY[self.goodpix[ip], i]
                        + np.conj(self.alm_tt[l,m]*self.NinvY[self.goodpix[ip], i]))
                p_r3[ip+self.mp] += clte[l]/cltt[l] * self.wl[l] * (self.alm_tt[l,m]*self.NinvY[self.goodpix[ip]+self.npix, i]
                        + np.conj(self.alm_tt[l,m]*self.NinvY[self.goodpix[ip]+self.npix, i]))
                i += 1
        return p_r3
        

    def TT_TE_covariance(self,cltt,clte):
        ell_tt = np.arange(self.ttmin,self.ttmax+1)
        self.tttt = 2.*(cltt[self.ttmin,self.ttmax+1]+self.ntt[:self.ttmax+1])**2/((2*ell+1.)\
                    *self.fskytt[:self.ttmax+1]**2)

        ell_te = np.arange(self.temin,self.temax+1)
        self.tete = ((cltt[self.temin:self.temax+1]+self.ntt_te[:self.temax+1])\
            *(clee[self.temin:self.temax+1]+self.nee_te[:self.temax+1])+clte[self.temin:self.temax+1]**2)\
            /((2.*ell_te+1.)*self.fskyte[:self.temax+1])
        self.ttte = 2. * ((cltt[self.temin:self.temax+1]+self.ntt[:self.temax+1])*clte[self.temin:self.temax+1])\
            /((2.*ell_te+1.)*self.fskyte[:self.temax+1]*self.fskytt[:self.temax+1])


    def initialize_gibbs_for_tt(self):
        gibbs_chain = fits.open(os.path.normpath(os.path.join(self.data_folder,self.gibbs_sigma_filename)))
        lmax = gibbs_chain[0].header['LMAX']
        numsamples = gibbs_chain[0].header['NUMSAMP']
        numchains = gibbs_chain[0].header['NUMCHAIN']
        numspec = gibbs_chain[0].header['NUMSPEC']

        data = gibbs_chain[0].data.T
        sigmas = data[:lmax,0,:numchains,:numsamples]

        cl_file = os.path.normpath(os.path.join(self.data_folder,
                                                                'lowlT/gibbs/test_cls.dat'))

        self.cl_tt_fiducial = np.loadtxt(cl_file,usecols=(1),unpack=True)[self.ttmin:self.gibbs_ell_max]

        self.lmax_br = len(sigmas[self.ttmin:self.gibbs_ell_max, 1, 1]) + self.ttmin - 1
        self.sigmas_br = sigmas[self.ttmin:self.lmax_br+1,:numchains+1,self.gibbs_first_iteration-1:self.gibbs_last_iteration+1:self.gibbs_skip]
        self.numchains_br = len(self.sigmas_br[self.ttmin,:,1])
        self.numsamples_br = len(self.sigmas_br[self.ttmin,1,:])


        offset = - 1.6375e30
        self.offset = self.compute_br_estimator(self.cl_tt_fiducial,offset)

        if (self.offset <-1.637e30):
            print("Error: offset in br_mod_dist not being computed properly",
                  "lmin = %s"%self.ttmin,
                  "lmax = %s"%self.lmax_br,
                  "numchain = %s"%self.numchains_br,
                  "numsamples = %s"%self.numsamples_br,
                  "offset = %s"%self.offset)


    def compute_br_estimator(self,cl,offset=None):
        ellfactor = np.arange(self.ttmin,self.lmax_br+1,dtype=float)
        if offset:
            tmp = offset
            subtotal = 0.
            for l in range(0,self.lmax_br-self.ttmin+1):
                x = self.sigmas_br[l, :, :] / float(cl[l])
                subtotal += 0.5*(2.*ellfactor[l]+1.)*(-x+np.log(x)) - np.log(self.sigmas_br[l, :, :])
            offset = max(offset, np.amax(subtotal))

            return offset
        else:
            lnL = 0.
            subtotal = 0.
            for l in range(0,self.lmax_br-self.ttmin+1):
                x = self.sigmas_br[l, :, :] / float(cl[l])
                subtotal += 0.5*(2.*ellfactor[l]+1.)*(-x+np.log(x)) - np.log(self.sigmas_br[l, :, :])

            lnL += np.sum(np.exp(subtotal-self.offset),dtype=float)

            if lnL > 1e-20:
                lnL = np.log(lnL)
            else:
                lnL = np.log(1e-30)
            return lnL

    
    def initialize_lowl_pol_data(self):
        teeebb_maskfile = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/mask_r3_p06_jarosik.fits'))
        alm_tt_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/alm_tt_fs_r9_ilc_nopixwin_9yr.dat'))
        wl_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'healpix_data/pixel_window_n0008.txt'))
        NinvYe_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/masked_ninvy_e_qu_r3_corrected_9yr.KaQV.fits'))
        NinvYb_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/tbeb/masked_ninvy_b_qu_r3_corrected_9yr.KaQV.fits'))
        ninvplninv_ee_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/masked_ee_ninvplninv_qu_r3_corrected_9yr.KaQV.fits'))
        ninvplninv_bb_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/masked_bb_ninvplninv_qu_r3_corrected_9yr.KaQV.fits'))
        ninvplninv_eb_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/tbeb/masked_eb_ninvplninv_qu_r3_corrected_9yr.KaQV.fits'))
        ninvplninv_be_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/tbeb/masked_be_ninvplninv_qu_r3_corrected_9yr.KaQV.fits'))
        NinvQUr3_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/masked_ninv_qu_r3_corrected_9yr.KaQV.fits'))
        map_q_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/wt_r3_9yr.KaQV.map_q'))
        map_u_file = os.path.normpath(os.path.join(self.data_folder,
                                                        'lowlP/std/wt_r3_9yr.KaQV.map_u'))

        Mask_R3 = fits.open(teeebb_maskfile)[1].data.field(1)
        self.goodpix = np.where(Mask_R3!=0)[0] 
        self.mp = len(self.goodpix)

        self.wl = np.loadtxt(wl_file,unpack=True)

        self.alm_tt = np.zeros((self.nlmax+1,self.nlmax+1),dtype=complex)
        
        l1,l2 = [],[]
        with open(alm_tt_file,'r') as file:
            for line in file:
                l1 += [float(line.split(',')[0][2:])]
                l2 += [float(line.split(',')[1][:-2])]

        i = 0
        for l in range(0,self.nlmax+1):
            for m in range(0,l+1):
                if np.sign(l2[i]) == -1:
                    self.alm_tt[l,m] = complex(str(l1[i])+'-'+str(abs(l2[i]))+'j')
                else:
                    self.alm_tt[l,m] = complex(str(l1[i])+'+'+str(abs(l2[i]))+'j') 
                i+=1

        #Reading N^{-1}P_l N^{-1} at res3
        if self.options['use_lowTBEB']:
            ninvplninv_tmp = np.zeros((self.nlmax+1, 2*self.npix, 2*self.npix, 4))
            self.ninvplninv = np.zeros((self.mp*(2*self.mp+1), 4*(self.nlmax-1)))
            
            ninvplninv_tmp[:,:,:,0] = fits.open(ninvplninv_ee_file)[0].data.T
            ninvplninv_tmp[:,:,:,1] = fits.open(ninvplninv_bb_file)[0].data.T
            ninvplninv_tmp[:,:,:,2] = fits.open(ninvplninv_eb_file)[0].data.T
            ninvplninv_tmp[:,:,:,3] = fits.open(ninvplninv_be_file)[0].data.T

            k = 0 
            for j in range(2*self.mp):
                for i in range(j):
                    ip = ngood[np.mod(i, self.mp)] + self.npix * (i/self.mp)
                    jp = ngood[np.mod(j, self.mp)] + self.npix * (j/self.mp)
                    self.ninvplninv[k,:self.nlmax] = ninvplninv_tmp[2:,ip,jp,0]
                    self.ninvplninv[k,self.nlmax-1:2*(self.nlmax-1)] = ninvplninv_tmp[2:,ip,jp,1]
                    self.ninvplninv[k,2*(self.nlmax-1):3*(self.nlmax-1)] = ninvplninv_tmp[2:,ip,jp,2]
                    self.ninvplninv[k,3*(self.nlmax-1):4*(self.nlmax-1)] = ninvplninv_tmp[2:,ip,jp,3]
                    k += 1
            #Reading N^{-1}Y
            NinvYe_tmp = fits.open(NinvYe_file)[0].data.T
            NinvYb_tmp = fits.open(NinvYb_file)[0].data.T
            self.NinvYe = np.zeros_like(NinvYe_tmp[0],dtype=complex)
            self.NinvYb = np.zeros_like(NinvYb_tmp[0],dtype=complex)
            for i in range(np.shape(NinvYe_tmp[0])[0]):
                for j in range(np.shape(NinvYe_tmp[0])[1]):
                    self.NinvYe[i,j] = complex(NinvYe_tmp[0][i,j],NinvYe_tmp[1][i,j])
                    self.NinvYb[i,j] = complex(NinvYb_tmp[0][i,j],NinvYb_tmp[1][i,j])

        else:
            ninvplninv_tmp = np.zeros((self.nlmax+1, 2*self.npix, 2*self.npix, 2))
            self.ninvplninv = np.zeros((self.mp*(2*self.mp+1), 2*(self.nlmax-1)))

            ninvplninv_tmp[:,:,:,0] = fits.open(ninvplninv_ee_file)[0].data.T
            ninvplninv_tmp[:,:,:,1] = fits.open(ninvplninv_bb_file)[0].data.T

            k = 0 
            for j in range(2*self.mp):
                for i in range(j):
                    ip = self.goodpix[np.mod(i, self.mp)] + self.npix * int(i/self.mp)
                    jp = self.goodpix[np.mod(j, self.mp)] + self.npix * int(j/self.mp)
                    self.ninvplninv[k,:self.nlmax-1] = ninvplninv_tmp[2:,ip,jp,0]
                    self.ninvplninv[k,self.nlmax-1:2*(self.nlmax-1)] = ninvplninv_tmp[2:,ip,jp,1]
                    k += 1
            #Reading N^{-1}Y
            NinvY_tmp = fits.open(NinvYe_file)[0].data.T
            self.NinvY = np.zeros_like(NinvY_tmp[0],dtype=complex)
            for i in range(np.shape(NinvY_tmp[0])[0]):
                for j in range(np.shape(NinvY_tmp[0])[1]):
                    self.NinvY[i,j] = complex(NinvY_tmp[0][i,j],NinvY_tmp[1][i,j])
            

        #Reading N^{-1} at res3
        self.Dp0 = np.zeros((2*self.mp, 2*self.mp))
        NinvQUr3 = fits.open(NinvQUr3_file)[0].data.T

        pix = np.arange(self.mp)
        for i in pix:
            self.Dp0[pix,i] = NinvQUr3[self.goodpix[pix],self.goodpix[i]]
            self.Dp0[pix,self.mp+i] = NinvQUr3[self.goodpix[pix],self.npix+self.goodpix[i]]
            self.Dp0[self.mp+pix,i] = NinvQUr3[self.npix+self.goodpix[pix],self.goodpix[i]]
            self.Dp0[self.mp+pix,self.mp+i] = NinvQUr3[self.npix+self.goodpix[pix],self.npix+self.goodpix[i]]


        #Reading maps at res3
        map_q = fits.open(map_q_file)[1].data.field(0)
        map_u = fits.open(map_u_file)[1].data.field(0)

        self.w_r3 = np.zeros(2*self.mp)
        self.w_r3[pix] = map_q[self.goodpix[pix]]*Mask_R3[self.goodpix[pix]]
        self.w_r3[pix+self.mp] = map_u[self.goodpix[pix]]*Mask_R3[self.goodpix[pix]]


    def get_diag_terms(self):
        ttfilename = os.path.normpath(os.path.join(self.data_folder,
                            'highl/wmap_likelihood_inputs_tt.p4v6.wmap9.kq85.cinv_v3.dat'))

        tefilename = os.path.normpath(os.path.join(self.data_folder,
                            'highl/wmap_likelihood_inputs_te.p5_final.dat'))

        self.cltt_dat,self.ntt,self.fskytt = np.loadtxt(ttfilename,usecols=(1,2,3),unpack=True)[:self.ttmax+1]
        self.clte_dat,self.ntt_te,self.nee_te,self.fskyte = np.loadtxt(tefilename,usecols=(1,3,4,5),unpack=True)[:self.temax+1]

        if self.options['use_highl_TB']:
            tbfilename = os.path.normpath(os.path.join(self.data_folder,
                            'highl/wmap_likelihood_inputs_tb.p5_final.dat'))

            self.cltb_dat,self.ntt_tb,self.nbb_tb,self.fskytb = np.loadtxt(tbfilename,usecols=(1,3,4,5),unpack=True)[:self.temax+1]


    def get_offdiag_terms(self):
        ttofffilename = os.path.normpath(os.path.join(self.data_folder,
                            'highl/wmap_likelihood_inputs_tt_offdiag.p4v4.wmap9.kq85.cinv_v3.dat'))
        teofffilename = os.path.normpath(os.path.join(self.data_folder,
                            'highl/wmap_likelihood_inputs_te_offdiag.p5_final.dat'))

        tmp = np.loadtxt(ttofffilename,usecols=(2,3),unpack=True)
        self.R_off_tttt = np.zeros((self.ttmax-self.ttmin+1,self.ttmax-self.ttmin+1))
        self.epsilon = np.zeros((self.ttmax-self.ttmin+1,self.ttmax-self.ttmin+1))
        for i in range(self.ttmax-self.ttmin+1):
            for j in range(i,self.ttmax-self.ttmin+1):
                self.epsilon[i,j] = tmp[0][j-1]
                self.epsilon[j,i] = self.epsilon[i,j]
                self.R_off_tttt[i,j] = tmp[1][j-1]
                self.R_off_tttt[j,i] = self.R_off_tttt[i,j]

        tmp = np.loadtxt(teofffilename,usecols=(2),unpack=True)
        self.R_off_tete = np.zeros((self.temax-self.temin+1,self.temax-self.temin+1))
        for i in range(self.temax-self.temin+1):
            for j in range(i,self.temax-self.temin+1):
                self.R_off_tete[i,j] = tmp[j-1]
                self.R_off_tete[j,i] = self.R_off_tttt[i,j]


    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": {cl: 2000 for cl in self.use_cl}}


    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=False)
        if self.options['use_lowTBEB']:
            return self.loglike(Cls.get("tt"),Cls.get("te"), Cls.get("ee"),
                                Cls.get("bb"),Cls.get("eb"), Cls.get("tb"))
        else:
            return self.loglike(Cls.get("tt"),Cls.get("te"), Cls.get("ee"),Cls.get("bb"))





