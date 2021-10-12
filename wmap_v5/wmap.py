"""
.. module:: wmap_v5
:Author: Umberto Natale

"""
import os
from typing import Optional

import numpy as np
from astropy.io import fits
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError


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
    gibbs_last_iteration: Optional[int] = 120000
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

        if self.options['use_lowl_TT'] and self.options['use_gibbs']:
            self.initialize_gibbs_for_tt()

        
        self.log.info("Initialization done!")


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

        cl_tt = np.loadtxt(cl_file,usecols=(1),unpack=True)[self.ttmin:self.gibbs_ell_max]

        lmax_br = len(sigmas[self.ttmin:self.gibbs_ell_max, 1, 1]) + self.ttmin - 1
        gibbs_idx = np.arange(self.gibbs_first_iteration,self.gibbs_last_iteration,self.gibbs_skip)
        sigmas_br = sigmas[self.ttmin:lmax_br,:numchains,gibbs_idx]

        offset = - 1.6375e30
        subtotal = 0
        for i in range(numsamples):
            for j in range(numchains):
                for l in range(0,lmax_br-self.ttmin+1):
                    x = sigmas_br[l, j, i] / cl_tt[l]
                    subtotal += 0.5*(2.*l+1.)*(-x+np.log(x)) - np.log(sigmas_br[l, j, i])

        offset = max(offset,subtotal)

        if (offset <-1.637e30):
            print("Error: offset in br_mod_dist not being computed properly",
                  "lmin = %s"%self.ttmin,
                  "lmax = %s"%lmax_br,
                  "numchain = %s"%numchains,
                  "numsamples = %s"%numsamples,
                  "offset = %s"%offset)

        lnL = 0.
        for i in range(numsamples):
            for j in range(numchains):
                subtotal = 0.
                for l in range(0,lmax_br-self.ttmin+1):
                    x = sigmas_br[l, j, i] / cl_tt[l]
                    subtotal += 0.5*(2.*l+1.)*(-x+np.log(x)) - np.log(sigmas_br[l, j, i])
                lnL = lnL + np.exp(subtotal-offset)

        if lnL > 1e-20:
            lnL = np.log(lnL)
        else:
            lnL = np.log(1e-30)
        
    


    def initialize_lowl_pol_data(self):
        '''
        This use lowl_TBEB by default. Likelihood without TBEB to be implemented...
        '''
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
        goodpix = np.where(Mask_R3!=0)[0] 
        mp = len(goodpix)

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

        #Reading N^{-1}Y
        NinvYe = fits.open(NinvYe_file)[0].data.T
        NinvYb = fits.open(NinvYb_file)[0].data.T

        #Reading N^{-1}P_l N^{-1} at res3
        self.ninvplninv = np.zeros((self.nlmax+1, 2*self.npix, 2*self.npix, 4))
        
        self.ninvplninv[:,:,:,0] = fits.open(ninvplninv_ee_file)[0].data.T
        self.ninvplninv[:,:,:,1] = fits.open(ninvplninv_bb_file)[0].data.T
        self.ninvplninv[:,:,:,2] = fits.open(ninvplninv_eb_file)[0].data.T
        self.ninvplninv[:,:,:,3] = fits.open(ninvplninv_be_file)[0].data.T

        #Reading N^{-1} at res3
        self.Dp0 = np.zeros((2*mp, 2*mp))
        NinvQUr3 = fits.open(NinvQUr3_file)[0].data.T

        pix = np.arange(mp)
        for i in pix:
            self.Dp0[pix,i] = NinvQUr3[goodpix[pix],goodpix[i]]
            self.Dp0[pix,mp+i] = NinvQUr3[goodpix[pix],self.npix+goodpix[i]]
            self.Dp0[mp+pix,i] = NinvQUr3[self.npix+goodpix[pix],goodpix[i]]
            self.Dp0[mp+pix,mp+i] = NinvQUr3[self.npix+goodpix[pix],self.npix+goodpix[i]]


        #Reading maps at res3
        map_q = fits.open(map_q_file)[1].data.field(0)
        map_u = fits.open(map_u_file)[1].data.field(0)

        self.w_r3 = np.zeros(2*mp)
        self.w_r3[pix] = map_q[goodpix[pix]]*Mask_R3[goodpix[pix]]
        self.w_r3[pix+mp] = map_u[goodpix[pix]]*Mask_R3[goodpix[pix]]


        def get_requirements(self):
            # State requisites to the theory code
            return {"Cl": {cl: self.lmax for cl in self.use_cl}}


        def loglike(self, dlte, dlee, **params_values):
            print(self.w_r3)


        def logp(self, **data_params):
            Cls = self.provider.get_Cl(ell_factor=True)
            return self.loglike(Cls.get("te"), Cls.get("ee"), **data_params)








