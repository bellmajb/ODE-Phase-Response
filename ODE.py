#############################################
# This is the definition of the ODE class.  #
#############################################

from __future__ import print_function
import sympy
from sympy.utilities.lambdify import lambdify
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import argrelextrema
from numpy import linalg as LA
import time

class ODE:
    'Common base class for all ODEs'
    ###################################################################
    # The initializer of the ODE class requires a string name (name), # 
    # a matrix of sympy variables (variables), a dictionary of sympy  #
    #    parameters and their values (parameters), a list of sympy    #
    #  parameters that act as placeholders for constant pulses to RHS #
    # functions (par_add), a matrix of sympy ODE RHS functions (RHS), #
    #  and values of the variables on the limit cycle at the maximum  #
    #                        value of frq mRNA.                       #
    ###################################################################
    def __init__(self, name, variables, parameters, par_add,
                 RHS, lim_cyc_vals):
        
        # Initialize all inputs
        self.name = name
        self.save_name = self.name.replace(" ", "")
        self.variables = variables
        self.num_vars = len(variables)
        self.parameters = parameters
        self.num_parms = len(parameters)
        self.par_add = par_add
        self.RHS = RHS
        self.lim_cyc_vals = lim_cyc_vals
        
        # All models will have the time variable t.      
        self.t = sympy.Symbol('t')

        #######################################################
        #  Create numpy functions, not symbolic, for the RHS  #
        #  as well as the jacobian wrt the variables as well  #
        #                 as the parameters                   #
        #######################################################

        # RHS sympy func to numpy func        
        self.f_ODE_RHS = self.RHStoFunc()
        
        # jacobian of RHS sympy func (wrt variables) to numpy func
        self.f_jac = self.RHStoJacobian(self.variables)
        # jacobian of RHS sympy func (wrt parameters) to numpy func
        self.f_dYdp = self.RHStoJacobian(self.parameters.keys())
        
        # Find the iPRCs (Q_interp) of the ODE along with associated
        # solutions (Y_interp), both necessary for PRC calculation.                  
        [self.Q_interp, self.Y_interp]  = self.findQ_interpAndY_interp()
        
        # Find the limit cycle solution y_gamma on a very fine grid
        # as well as the period to be use with dirPRC calculation.
        step_size = 10e-6 
        self.period = self.findPeriod(step_size)
        [self.t_gamma, self.y_gamma] = self.findGammaSol(step_size)              
    
    #############################################################    
    #############################################################
    ######                                                #######
    ######  METHODS FOR THE ODE CLASS (ALPHABETIC ORDER)  #######  
    ######                                                #######
    #############################################################
    #############################################################

#---------------------------------------------------------------------

    ####################################################################
    # ApproxPRC computes an approximate PRC using the iPRCs (Q_interp) #
    #       of an ODE given a vector of phases at which to pulse       #
    # (time_points), a dictionary of symbolic parameters to be pulsed  #
    # with their corresponding amplitudes (pulses), and the length of  #
    #                       the pulse (sigma).                         #
    ####################################################################
    
    # PRC pulsing parameter with index p_ind and amplitude amp
    def approxPRC(self,time_points,pulses,sigma):
        
        # This function just uses Q_interp to determine a PRC in response
        # to an additive pulse of a RHS function. Input the index of the
        # variable that the RHS is being added to, and the amplitude of
        # the pulse.
        def PRCAdd(p_ind,amp):   
            
            prc_add_out = sigma*amp*self.Q_interp(time_points)[:,p_ind]
            
            return prc_add_out
        
        # This function uses a convolution integral to determine a PRC in 
        # response to a pulse of specific parameters. Input a dictionary of
        # parameters to be pulsed and their corresponding amplitudes.        
        def PRCPar(par_pulses):

            # Initialize the prc yielded from pulsing these indicies            
            prc_par_out = np.zeros(time_points.shape)    
        
            # pulse def 
            def dp(u):
                
                pulse_start = 0.
                pulse_end = sigma
                
                # Only on the interval of the pulse, dp is a function that
                # returns an array of 0s for indicies corresponding to non
                # pulsed parameters, and the amplitude of the pulse for the
                # indicies of the pulsed parameters                 
                if (u > pulse_start) & (u < pulse_end):
                    
                    # Initialize output.                    
                    y = np.zeros((self.num_parms))                    
                    
                    for this_par in par_pulses:
                        this_amp = par_pulses[this_par]
                        this_ind = self.parameters.keys().index(this_par)
                        y[this_ind] = this_amp
                
                else:
                    
                    y =  np.zeros(self.num_parms)
                
                return y     
            
            for ind, val in enumerate(time_points):

                #integrand for PRC ( ( Q dot Jac_p(Y) ) * dp )
                def f(u):
                    
                    Q =  self.Q_interp(u + val)
                    Jac_p = self.f_dYdp(self.Y_interp(u + val),
                                        u, self.parameters.values())
                    y = np.dot(np.dot(Q,Jac_p), dp(u))
    
                    return y
                
                shift, err = integrate.quad(f, 0, sigma)
                            
                prc_par_out[ind] = shift                
                
            return prc_par_out

        ###############################################################
        # Now we will find the PRC using the above 2 functions. First #
        #       we will add all of the PRCs from pulsing additive     #
        #   constants using PRCAdd, and with the remaining indicies   #
        # we can find the PRC from pulsing the rest of the parameters #
        #                   all at once using PRCPar.                 #
        ###############################################################       
        
        # Initialize the PRC
        prc_vec = np.zeros(time_points.shape)
        
        # Initialize a dictionary of pulses (parameters and correspoinding
        # amplitudes) that affect parameters as opposed to par_add
        # parameters.
        par_pulse_dic = {} 

        # Determine the indicies of parameters in parameters and par_add
        # affected by the pulse and add these prcs to prc_vec. If the current
        # parameter is in parameters, add it and the corresponding amp to
        # par_pulse_dic.
        for this_par in pulses:
            
            # Name the current pulse amplitude.            
            this_amp = pulses[this_par]
            
            # If this parameter is in par_add, add the PRC to prc_vec
            if (this_par in self.par_add) == True:                
                # Need to input the index corresponding to par_add                
                this_ind = self.par_add.index(this_par)
                prc_vec += PRCAdd(this_ind,this_amp)
            
            # If not in par_add, it affects a parameter. Add the parameter
            # and the corresponding amplitude of this parameter pulse to 
            # the dictionary par_pulse_dic.
            elif (this_par in self.parameters.keys()) == True:
                
                par_pulse_dic[this_par] = this_amp
            
            # Tell the user if this parameter was entered incorrectly.
            else:
                
                str1 = str(this_par)
                str2 = ' is not an acceptable parameter'
                print(''.join([str1, str2]))
        
        # This means there is at least one pulsing parameter in pulses
        # affecting parameters as opposed to par_add. Add the PRC resulting
        # from the pulses affecting these parameters to prc_vec.
        if (not par_pulse_dic) == False:
            
            prc_vec += PRCPar(par_pulse_dic)
            
        # Lastly, mod out correctly to ensure the PRC is in the interval 
        # [-period/2, period/2]
        prc_vec_out = (prc_vec + self.period/2.) % \
                      (self.period) - self.period/2.        
        
        return prc_vec_out 
 
#---------------------------------------------------------------------
    def changePars(self, pulses_pars):

        # change parameters for the pulse. Don't forget to change them back
        for this_par in pulses_pars:
            
            this_amp = pulses_pars[this_par]                
            
            # Add the amp of the pulse to the parameter
            self.parameters[this_par] += this_amp    

#---------------------------------------------------------------------
    
    def addParsToRHS(self, pulses_add):

        # In the RHS function, incorporate all additive pulses.
        def f_RHS_add(Y,t,p):
           
            # The unpulsed RHS (in terms of additive pulses):                
            a = self.f_ODE_RHS(Y,t,p)
            
            # A vector to be filled with corresponding additive pulses:
            b = np.zeros(a.shape)
            for this_par in pulses_add:
                this_par_amp = pulses_add[this_par]
                this_par_ind = self.par_add.index(this_par)
                b[this_par_ind] = this_par_amp
        
            return a+b
            
        return f_RHS_add
        
#---------------------------------------------------------------------
   
    def changeParsBack(self, pulses_pars):
        
        # set parameters back to normal for after pulses
        for this_par in pulses_pars:
            
            this_amp = pulses_pars[this_par]                
            
            # Return the parameter to its original value
            self.parameters[this_par] -= this_amp      

#---------------------------------------------------------------------        

    #################################################################
    #      approxPRC2 computes an approximate PRC using approxPRC   #
    # but adding the following additonal term to increase accuracy: #
    #            int_{\phi}^{\phi+sigma) Q(s) \cdot ...             #
    #       [D_xf(x_{\gamma}(s),p)\epsilon_x delx(\phi,s)]ds        #
    #     As with approxPRC, approxPRC2 inputs (time_points), a     #
    #        dictionary of symbolic parameters to be pulsed         #
    # with their corresponding amplitudes (pulses), and the length  #
    #                    of the pulse (sigma).                      #
    #################################################################
    
    # PRC pulsing parameter with index p_ind and amplitude amp
    def approxPRC2(self,time_points,pulses,sigma):
        
        # approxPRC is computed first to be added to the additional
        # terms later
        prc_vec_out_1 = self.approxPRC(time_points,pulses,sigma)
        
        plt.plot(time_points, prc_vec_out_1)
        plt.show()
        
        # Initialize the second addition to be computed as another integral
        prc_vec_out_2 = np.zeros(prc_vec_out_1.shape)        
        
        # Do the other integral here
        
        for ind, val in enumerate(time_points):        

            #integrand for PRC ( ( Q dot Jac_x(Y) ) * delx )
            def f(u):
                
                Q = self.Q_interp(u + val)
                Jac_x = self.f_jac(self.Y_interp(u + val),
                                   u, self.parameters.values())
                delx = self.find_delx(val,pulses,sigma)
                
                y = np.dot(np.dot(Q,Jac_x), delx(u))
    
                return y
            
            shift, err = integrate.quad(f, 0, sigma)
                        
            prc_vec_out_2[ind] = shift
        
        plt.plot(time_points, prc_vec_out_2)
               
        prc_vec_out = prc_vec_out_1 + prc_vec_out_2              
        
        return prc_vec_out

#---------------------------------------------------------------------

    # Here is a function that returns an interpolation of delx
    # for an input phi by solving x' = f(x,p+dp) with initial conditions
    # x(0) = 
    def find_delx(self,phi,pulses,sigma):
        
        ###############################################################
        # Change the parameter values during the pulse and integrate. #
        ###############################################################
        
        # Divide the pulses dictionary into two dictionaries, one for
        # parameters pulses, one for additive pulses.
        [parameters_pulses, par_add_pulses] = self.dividePulses(pulses)
        
        # Change the parameters for the pulse
        self.changePars(parameters_pulses)
        
        # Add to RHS where necessary
        f_ODE_RHS_add_pulse = self.addParsToRHS(par_add_pulses)           
        
        # Define times for integration during the pulse (dp):
        dp_int_len = sigma
        dt = 0.005
        num_grid_points = 100.*int(dp_int_len/dt+1)           
        dp_times = np.linspace(0,dp_int_len,num_grid_points)            
        
        # Initial conditions for delx are the same as Y_interp at phi            
        dp_init = np.squeeze(self.Y_interp(phi))
 
        # Integrate during the pulse (dp):
        dp_sol = integrate.odeint(f_ODE_RHS_add_pulse, dp_init,
                                  dp_times, (self.parameters.values(),))
                        
        plt.plot(dp_times, dp_sol)
        plt.show()
        
        # Interpolate the pulsed solution per variable (put in list)
        Y_pulsed_interp_list = [InterpolatedUnivariateSpline(dp_times, dp_sol[:,ind])\
                                for ind in range(0,self.num_vars)]
        
        # Use the list of the interpolated pulsed solution to define an
        # interpolation function that returns an np array.                            
        def Y_pulsed_interp(times):
            output = Y_pulsed_interp_list[0](times)
            for ind in range (1,self.num_vars):
                output = np.vstack((output,Y_pulsed_interp_list[ind](times)))
            return output.transpose()
        
        # Return the parameters to their original values (unpulse them)
        self.changeParsBack(parameters_pulses)            
        
        def delx(times):
            
            diff = Y_pulsed_interp(times) - self.Y_interp(times)
            
            return np.squeeze(diff)
            
        return delx

#---------------------------------------------------------------------

   ###############################################################
   # This method compares the direct PRC with experimental data. #
   # Input the x and y values for the experimental data, as well #
   #   as the pulse and the pulse length for the direct PRC and  #
   # the method calculates the direct PRC and plots it with the  #
   #  experimental PRC. It also calculates and prints the NRMSE  #
   #      of the direct PRC as an approximation of the data.     #
   ###############################################################    
    def compDirAndData(self, data_x, data_y, pulses, sigma):
        
        # A function for caculating the NRMSE
        def findNRMSE(v_sim, v_true):
            
            MSE = np.mean((v_sim-v_true)**2)
            RMSE = MSE**(0.5)
            range_v_true = max(v_true)-min(v_true)
            # Normalize by the range of the 'true' data.            
            NRMSE = (1./range_v_true)*RMSE
            
            return NRMSE 
        
        # Calculate the direct PRC
        dirPRC = self.dirPRC(data_x, pulses, sigma)
        
        # Find the NRMSE of the direct PRC compared to the data
        NRMSE = findNRMSE(dirPRC, data_y)
        
        # Print the NRMSE
        str1 = 'NRMSE = '
        str2 = str(NRMSE)
        print(''.join([str1, str2]))
        
        # Plot the PRCs
        plt.plot(data_x, dirPRC, 'go', label = 'Direct PRC')
        plt.plot(data_x, data_y, label = 'Experimental PRC')
        
        # Label the figure.
        plt.xlabel('Time of Pulse (h)', fontsize = 14)
        plt.ylabel('Advance/Delay (h)', fontsize = 14)
        str1 = 'Direct PRC vs Data for \n'
        str2 = self.name
        str3 = ''.join([str1, str2])
        plt.title(str3, fontsize = 20)
        plt.legend(loc='best')
        plt.show()
        
#---------------------------------------------------------------------

   ########################################################################
   # This method compares the direct and iPRC methods of calculating the  #
   # PRC numerically. Inputting the phases at which to calculate the PRC, #
   # the pulse, and the pulse length, the method calculates the PRCs and  #
   # plots them together. It also calculates and prints the NRMSE of the  #
   #         iPRC method PRC as an approximation of the direct PRC.       #
   ########################################################################    
    def compDirAndDirOld(self, phases, pulses, sigma, Tol):
        
        # A function for caculating the NRMSE
        def findNRMSE(v_sim, v_true):
            
            MSE = np.mean((v_sim-v_true)**2)
            RMSE = MSE**(0.5)
            range_v_true = max(v_true)-min(v_true)
            # Normalize by the range of the 'true' data.            
            NRMSE = (1./range_v_true)*RMSE
            
            return NRMSE        
        
        # Calculate the two PRCs.
        t_1_old = time.time()
        dir_PRC_old = self.dirPRCOld(phases, pulses, sigma)
        t_2_old = time.time()
        t_tot_old = t_2_old - t_1_old
        print('Elapsed time for old method = {} seconds'.format(t_tot_old))
        
        t_1_new = time.time()
        dir_PRC = self.dirPRC(phases, pulses, sigma, Tol)
        t_2_new = time.time()
        t_tot_new = t_2_new - t_1_new
        print('Elapsed time for new method = {} seconds'.format(t_tot_new))        
        
        # Calculate the decrease in time
        t_dec = t_tot_old - t_tot_new
        print('The new method is {} seconds faster'.format(t_dec))
        
        # Calculate the NRMSE of the iPRC approximation.
        this_NRMSE = findNRMSE(dir_PRC_old,dir_PRC)
        
        # Output the NRMSE.
        print('NRMSE = {}'.format(this_NRMSE))
        
        # Plot the PRCs
        plt.plot(phases, dir_PRC_old, 'go', label = 'Old Direct Method')
        plt.plot(phases, dir_PRC, label = 'New Direct Method')
        
        # Label the figure.
        plt.xlabel('Time of Pulse (h)', fontsize = 14)
        plt.ylabel('Advance (h)', fontsize = 14)
        str1 = 'New vs Old Direct PRC for \n'
        str2 = self.name
        str3 = ''.join([str1, str2])
        plt.title(str3, fontsize = 20)
        plt.legend(loc='best')
        
        if len(pulses) == 1:        
            
            path = 'OldVsNewDirPRCPLots/'
            filename = ''.join([path,
                                self.save_name,
                               '_OldVsNewDirPRC_',
                               str(pulses.keys()[0]),
                               '_',
                               str(pulses.values()[0]),
                               '.png'])
            plt.savefig(filename, bbox_inches='tight')
        
        plt.show()

#---------------------------------------------------------------------

   ########################################################################
   # This method compares the direct and iPRC methods of calculating the  #
   # PRC numerically. Inputting the phases at which to calculate the PRC, #
   # the pulse, and the pulse length, the method calculates the PRCs and  #
   # plots them together. It also calculates and prints the NRMSE of the  #
   #         iPRC method PRC as an approximation of the direct PRC.       #
   ########################################################################    
    def compDirAndiPRC(self, phases, pulses, sigma):
        
        # A function for caculating the NRMSE
        def findNRMSE(v_sim, v_true):
            
            MSE = np.mean((v_sim-v_true)**2)
            RMSE = MSE**(0.5)
            range_v_true = max(v_true)-min(v_true)
            # Normalize by the range of the 'true' data.            
            NRMSE = (1./range_v_true)*RMSE
            
            return NRMSE        
        
        # Calculate the two PRCs.        
        iPRC = self.approxPRC(phases, pulses, sigma)
        dirPRC = self.dirPRC(phases, pulses, sigma)
        
        # Calculate the NRMSE of the iPRC approximation.
        this_NRMSE = findNRMSE(iPRC,dirPRC)
        
        # Output the NRMSE.
        str1 = 'NRMSE = '
        str2 = str(this_NRMSE)
        print(''.join([str1,str2]))
        
        # Plot the PRCs
        plt.plot(phases, iPRC, 'go', label = 'iPRC Method')
        plt.plot(phases, dirPRC, label = 'Direct Method')
        
        # Label the figure.
        plt.xlabel('Time of Pulse (h)', fontsize = 14)
        plt.ylabel('Advance/Delay (h)', fontsize = 14)
        str1 = 'iPRC vs Direct PRC for \n'
        str2 = self.name
        str3 = ''.join([str1, str2])
        plt.title(str3, fontsize = 20)
        plt.legend(loc='best')
        
        if len(pulses) == 1:        
            
            path = 'CompDirAndiPRCPLots/'
            filename = ''.join([path,
                                self.save_name,
                               '_DirVsiPRC_',
                               str(pulses.keys()[0]),
                               '_',
                               str(pulses.values()[0]),
                               '.png'])
            plt.savefig(filename, bbox_inches='tight')
        
        plt.show()

#---------------------------------------------------------------------   
   
    # A destructor for the ODE class   
    def __del__(self):
        
        class_name = self.__class__.__name__
        print(class_name, "destroyed")    

#---------------------------------------------------------------------

    ###################################################################
    #  dirPRC computes an approximate PRC using the iPRCs (Q_interp)  #
    #       of an ODE given a vector of phases at which to pulse       #
    # (time_points), a dictionary of symbolic parameters to be pulsed  #
    # with their corresponding amplitudes (pulses), and the length of  #
    #                           the pulse.                             #
    ####################################################################
    
    # PRC pulsing parameter with index p_ind and amplitude amp
    def dirPRCOld(self,phases,pulses,sigma):       
        
        # Initialize the output 
        prc_out = np.zeros(phases.shape)        
        
        # First, the model needs to be integrated for a long enough amount
        # of time for a pulsed trajectory to be able to return to the LC.
        int_len = 750.
        dt = 0.005
        num_grid_points = int(int_len/dt+1)
        dir_times = np.linspace(0,int_len,num_grid_points)
        
        # Solve the unpulsed ODE
        Y_dir_sol = self.solve(self.lim_cyc_vals,dir_times)
        
        # Only save frq mRNA to be used as a reference for phase.        
        frq_mRNA_dir_sol = Y_dir_sol[:,0]
        
        # Find the indicies of frq mRNA local maximums.
        frq_mRNA_max_indicies = argrelextrema(frq_mRNA_dir_sol,np.greater)[0]
        
        # Save the last frq mRNA max as reference for phase when comparing
        # to a pulsed trajectory.
        last_frq_max_ind = frq_mRNA_max_indicies[-1]
        
        # The time of the last frq mRNA local max is the phase that the
        # pulsed solution will be compared to.        
        dir_max_time = dir_times[last_frq_max_ind]
        
        # Now calculate the phase shifts for pulses given at the various
        # phases (val)
        for ind, val in enumerate(phases):
        
            # Integrate up to the pulse; before pulse (bp)
            bp_int_len = val
            num_grid_points = int(bp_int_len/dt+1)
            bp_times = np.linspace(0,bp_int_len,num_grid_points)
            
            # Solve before the pulse            
            bp_sol = self.solve(self.lim_cyc_vals,bp_times)
            
            # Save end values as initial vals during pulse.
            bp_last = bp_sol[-1]
            
            ###############################################################
            # Change the parameter values during the pulse and integrate. #
            ###############################################################

            # Divide the pulses dictionary into two dictionaries, one for
            # parameters pulses, one for additive pulses.
            [parameters_pulses, par_add_pulses] = self.dividePulses(pulses)
            
            # Change the parameters for the pulse
            self.changePars(parameters_pulses)
            
            # Add to RHS where necessary
            f_ODE_RHS_add_pulse = self.addParsToRHS(par_add_pulses)           
            
            # Define times for integration during the pulse (dp):
            dp_int_len = sigma
            num_grid_points = 100.*int(dp_int_len/dt+1)
            #num_grid_points = int(dp_int_len/dt+1)            
            dp_times = np.linspace(0,dp_int_len,num_grid_points)
            
            # Integrate during the pulse (dp):
            dp_sol = integrate.odeint(f_ODE_RHS_add_pulse, bp_last,
                                      dp_times, (self.parameters.values(),))
                                      
            # Save end values as initial values for after pulse
            dp_last = dp_sol[-1] 
            
            # Return the parameters to their original values (unpulse them)
            self.changeParsBack(parameters_pulses)            
                
            # Define times for integration after the pulse ending at 1/2 a 
            # period after the unpulsed max
            ap_int_len = dir_max_time-bp_int_len-dp_int_len+self.period/2.
            num_grid_points = int(ap_int_len/dt+1)
            ap_times = np.linspace(0,ap_int_len,num_grid_points)
            
            # Integrate after the pulse (ap):            
            ap_sol = self.solve(dp_last, ap_times)
            
            # Only save frq mRNA to be used as the reference for phase.            
            frq_mRNA_ap_sol = ap_sol[:,0]
            
            # Find the indicies of frq mRNA local maximums (pulsed trajectory).
            frq_mRNA_ap_max_indicies = argrelextrema(frq_mRNA_ap_sol,
                                                     np.greater)[0]
            
            # Save the last frq mRNA max as reference for phase.
            last_frq_ap_max_ind = frq_mRNA_ap_max_indicies[-1]
            
            # The time of the last frq mRNA local max is the phase of the
            # pulsed solution.       
            ap_max_time = ap_times[last_frq_ap_max_ind]
            
            pulsed_max_time = ap_max_time + bp_int_len + dp_int_len            
            
            # The phase shift is simply the difference between the times
            # of the last frq mRNA local max.            
            prc_out[ind] = dir_max_time - pulsed_max_time
            
        # Lastly, mod out correctly to ensure the PRC is in the interval 
        # [-period/2, period/2]
        prc_out_scaled = (prc_out + self.period/2.) % \
                         (self.period) - self.period/2.        
        
        return prc_out_scaled

#---------------------------------------------------------------------

    ###################################################################
    #  dirPRC computes an approximate PRC using the iPRCs (Q_interp)  #
    #       of an ODE given a vector of phases at which to pulse       #
    # (time_points), a dictionary of symbolic parameters to be pulsed  #
    # with their corresponding amplitudes (pulses), and the length of  #
    #                           the pulse.                             #
    ####################################################################
    
    # PRC pulsing parameter with index p_ind and amplitude amp
    def dirPRC(self,phases,pulses,sigma,Tol):
        
        # First, find the limit cycle solution (lim_cyc_times, x_gamma):
        dt = 0.005     # This step size has been tested to ensure convergence
        int_start = 0.
        int_end = self.period
        int_len = int_end - int_start
        num_grid_points = int(int_len/dt + 2)        
        lim_cyc_times = np.linspace(int_start, int_end, num_grid_points)
#        x_gamma = self.solve(self.lim_cyc_vals, lim_cyc_times)
        
        # Define a function that finds the phase shift for a pulse given
        # at an input phase (phi).
        def findPhaseShift(phi):
            
            # Specify the tolerances for the distance of the pulsed trajectory
            # to the limit cycle (E_tol) and the total time to integrate 
            # before giving up (t_tol)
            E_tol = Tol
            t_tol = 50*self.period
            
            #######################################################
            # Pulse the parameters and find the pulsed trajectory #
            #    starting at t = phi going to t = phi + sigma     #
            #######################################################
            
            # Divide the pulses dictionary into two dictionaries, one for
            # parameters pulses, one for additive pulses.
            [parameters_pulses, par_add_pulses] = self.dividePulses(pulses)
            
            # Change the parameters for the pulse
            self.changePars(parameters_pulses)
            
            # Add to RHS where necessary
            f_ODE_RHS_add_pulse = self.addParsToRHS(par_add_pulses)           
            
            # Define times for integration during the pulse (dp):
            dp_int_start = 0
            dp_int_end = sigma            
            dp_int_len = dp_int_end - dp_int_start
            num_grid_points = 100.*int(dp_int_len/dt+1)
            dp_times = np.linspace(dp_int_start,dp_int_end,num_grid_points)
            
            # The initial values for the pulsed trajectory at phi dp_init
            # will be the same as without the pulse, x_gamma(phi). This can 
            # be found with Y_interp.
            dp_init = np.squeeze(self.Y_interp(phi))            
            
            # Integrate during the pulse (dp):
            dp_sol = integrate.odeint(f_ODE_RHS_add_pulse, dp_init,
                                      dp_times, (self.parameters.values(),))
                                      
            # Save end values of this integral (y_init). We will define y as
            # the solution of y' = f(y,p) starting at y_init for a time length
            # of one period. 
            y_init = dp_sol[-1]            
            
            # Return the parameters to their original values (unpulse them)
            self.changeParsBack(parameters_pulses)            
            
            # Only track the total time of the pulsed trajectory (t_tot) to be
            # used in the while loop ensuring the loop stops eventually.
            t_tot = phi + sigma          
            
            # Initialize the error above the tolerance to make sure it enters
            # the while loop and the pulsed trajectory gets to move forward
            # in time at least one period.
            E = E_tol + 1

            # Continually move one period in time on the pulsed trajectory
            # until the criteria are satisfied.
            while (E > E_tol) & (t_tot < t_tol):            
            
                y = self.solve(y_init, lim_cyc_times) 
                
                # Find the last value of the solution. This is the next
                # value on the isochron
                y_last = y[-1]            
                    
                # Update the error (E)
                dist = self.y_gamma - y_last  
                inf_dist = LA.norm(dist, np.inf, axis=1) 
                E = min(inf_dist) 
                E_ind = np.argmin(inf_dist)

                # Update t_tot
                t_tot += self.period                
                
                # Update y_init
                y_init = y_last

            # The limit cycle time of the final y_gamma that minimizes
            # inf_dist is the phase of the pulsed trajectory at the end
            # of the pulse
            phi_x = self.t_gamma[E_ind]          
            
            # The phase after the pulse of the unpurturbed solution is simply
            # the time at which the pulse was applied (phi) plus the time of
            # the pulse (sigma). This phase will be compared to the phase
            # of the pulsed solution directly after the pulse.
            phi_x_gamma = phi + sigma            
            
            # Solve for the phase shift and then return it
            shift = phi_x - phi_x_gamma
            
            return shift
        
        # Initialize the output prc vec
        prc_out_vec = np.zeros(phases.shape)
        
        # For each time point requested in phases, find the phase shift and 
        # add it to prc_out_vec before returning it
        for ind, val in enumerate(phases):
            
            prc_out_vec[ind] = findPhaseShift(val)
        
        prc_out_vec_scaled = (prc_out_vec + self.period/2.) % \
                             (self.period) - self.period/2.        
        
        return prc_out_vec_scaled

#---------------------------------------------------------------------
    
    def dividePulses(self,pulses):
        
        # Initialize empty dictionaries for parameter pulses and adding pulses.
        parameters_pulses = {}            
        par_add_pulses = {}            
        
        for this_par in pulses:
            
            this_amp = pulses[this_par]
            
            # If the current parameter is in par_add add it to the
            # par_add_pulses dictionary.                
            if (this_par in self.par_add) == True: 
                
                par_add_pulses[this_par] = this_amp
            
            # If the current parameter is in parameters, add it to
            # the parameters_pulses dictionary.                
            elif (this_par in self.parameters) == True:
                
                parameters_pulses[this_par] = this_amp
            
            # This means it must have not been entered correctly.
            else:

                str1 = str(this_par)
                str2 = ' is not a parameter.'
                print(''.join([str1, str2]))
                
        return parameters_pulses, par_add_pulses

#---------------------------------------------------------------------
    
    # Finds the limit cycle solution as well as the period of the ODE
    # used to define self.t_gamma, self.y_gamma, and self.period
    def findGammaSol(self, step_size):
        
        dt = step_size
        # This makes sure the step sizes are at most dt
        num_steps = int(self.period/dt + 2)
        times_gamma = np.linspace(0, self.period, num_steps)
        sol_gamma = self.solve(self.lim_cyc_vals, times_gamma)
        
        return times_gamma, sol_gamma 

#---------------------------------------------------------------------
    
    # Finds the period of the limit cycle solution of the model.
    def findPeriod(self, step_size):                               
       
        ############################################################       
        #  Integrating from the max of frq mRNA (at int_start) we  #
        # can find the next max to find the period. The Neurospora # 
        #  circadian clock period is around 22 hours, so integrate #
        #               to 24 to ensure a full period.             #
        ############################################################

        dt = step_size
        int_start = 0.
        int_end = 30.
        int_len = int_end - int_start
        # This makes sure the step sizes are at most dt
        num_grid_points = int(int_len/dt + 2)
        int_times = np.linspace(int_start,int_end,num_grid_points)
        
        # Find the frq mRNA solution.        
        frq_sol = self.solve(self.lim_cyc_vals,int_times)[:,0]
                                     
        # Now find the max, starting plenty far away from the initial max
        int_times_after_start = int_times[len(int_times)/2:]                                 
        frq_sol_after_start = frq_sol[len(int_times)/2:]
        frq_max_index = frq_sol_after_start.argmax()
        
        period = int_times_after_start[frq_max_index]    
        
        return period

#---------------------------------------------------------------------
    
    # Finds the iPRCs, returns Q_interp and Y_interp
    def findQ_interpAndY_interp(self):
        
        ########################################################
        # Find the solution of the ODE on a long interval and  #
        # interpolate it. The interpolated solution (Y_interp) #
        #  will be used to calculate the iPRCs (Q_interp) and  #
        #           the PRCs using the iPRC method             #
        ########################################################
        
        # Solve the ODE over a long time to ensure when reversing the
        # integration, the adjoint can return to the limit cycle
        t_start = 0.0
        t_end = 300.0
        num_steps = 300001
        ode_times = np.linspace(t_start, t_end, num_steps)
        
        # Solve
        Y_sol = self.solve(self.lim_cyc_vals,ode_times)

        # Store Final Values. These will help define initial values for
        # finding the iPRC
        Ylast = Y_sol[num_steps-1,:]    
        
        # For each variable interpolate the solution and add to a list
        Y_interp_list = [InterpolatedUnivariateSpline(ode_times, Y_sol[:,ind])\
                         for ind in range(0,self.num_vars)]        
        
        # For given times (times) Y_interp returns an np array with
        # columns of arrays of interpolate variables evaluated at times. 
        def Y_interp(times):
            output = Y_interp_list[0](times)
            for ind in range (1,self.num_vars):
                output = np.vstack((output,Y_interp_list[ind](times)))
            return output.transpose()
        
        ########################################################
        # Find the iPRCs of the ODE and return an interpolated #
        #                    them as Q_interp                  #
        ########################################################
        
        # Solve Adjoint Equation
        def adjoint_RHS(Y, t, p, interp_f):
            return - np.dot(self.f_jac(interp_f(t), t, p).T, Y)
            
        backward_times = np.linspace(t_end, t_start, num_steps)
        
        # The Adjoint solution Q has the unique property that Q dot dY = 1.
        # The following assignments ensure this property holds: we make Q0
        # be [0, 0, ... , 0, 1, Q0_last] where Q0_last then is calculated
        # to ensure the dot product holds.
        Q0 = [0.] * (self.num_vars-2)

        Q0.append(1.0)
        
        dY_2nd_last = self.f_ODE_RHS(Ylast, t_end,
                                     self.parameters.values())[self.num_vars-2]
        dY_last = self.f_ODE_RHS(Ylast, t_end,
                                 self.parameters.values())[self.num_vars-1]
        Q0_last = (1.-dY_2nd_last)/dY_last      
        Q0.append(Q0_last)
        
        # Integrate adjoint equation backwards in time
        Q = integrate.odeint(adjoint_RHS, Q0, backward_times,
                             (self.parameters.values(), Y_interp))
            
        # Create a list of interpolated Q solutions (per variable) to be
        # referenced when producing Q_interp        
        Q_interp_list = [InterpolatedUnivariateSpline(ode_times,Q[:,ind]) \
                         for ind in range(0,self.num_vars)]
        
        # Q_interp is our function which interpolates the solution to the
        # adjoint (the iPRCs). Notice the times need to be reversed because
        # Q was solved for using backwards_times.
        def Q_interp(times):
            output = Q_interp_list[0](t_end-times)
            for ind in range (1,self.num_vars):
                output = np.vstack((output,Q_interp_list[ind](t_end-times)))
            # Need to transpose in order to create a matrix with the right
            # dimensions, and squeeze to return a 1-D, not a 2-D array.
            return np.squeeze(output.transpose())
        
        # Q_interp is our function of interpolated iPRCs, and Y_interp is
        # our corresponding interpolated Y values needed when calculating
        # the PRC using the iPRC method.
        return Q_interp, Y_interp
        
#---------------------------------------------------------------------        
        
    # Converts the RHS input sympy func to a numpy func
    def RHStoFunc(self):
        
        f_dY = lambdify([self.variables, self.t,
                         self.parameters.keys()], self.RHS, use_array=True)
        
        # Wrapper around f_dY so it return a 1D, not 2D array
        def f_ODE_RHS(Y, t, p):
            return np.squeeze(f_dY(Y, t, p))
        
        return f_ODE_RHS
        
#---------------------------------------------------------------------
    
    # Converts the jacobian of the RHS input sympy func to a numpy func    
    def RHStoJacobian(self,wrt):
        
        return lambdify([self.variables, self.t,
                         self.parameters.keys()], self.RHS.jacobian(wrt),
                         use_array=True)
                         
#---------------------------------------------------------------------
      
    # Solves the ODE on the given time interval (ode_times) starting at
    # initial conditions Y0.
    def solve(self,x_init,ode_times):
        
        return integrate.odeint(self.f_ODE_RHS, x_init,
                                ode_times, (self.parameters.values(),))
                                
#---------------------------------------------------------------------
    
    # Plots the results from the solve method.                          
    def solveAndPlot(self,Y0,ode_times):        
        
        solution = self.solve(Y0,ode_times)
        plt.plot(ode_times,solution) 
        
#---------------------------------------------------------------------
        
    ###########################################################
    # This method determines a PRC for the ODE that minimizes #
    #   the sum of square errors (SSE) between the PRC and    #
    #   some experimental data. The inputs include x and y    #
    # values the PRC will be compared to (data_x, data_y), a  #
    # dictionary of parameters to be pulsed and corresponing  #
    #      initial guesses for amplitudes (pulse_pars),       #
    #   a dictionary of pulsed parameters and corresponding   #
    #  lower bounds (pulse_lows), and a dictionary of pulsed  #
    #         parameters and corresponding upper bounds       #
    # (pulse_highs). The boundary dictionaries are optional.  #
    #   The method uses perturbation analysis to approximate  #
    #      the PRC as a linear combination of unit PRCs.      #
    ###########################################################
    def optimalPRC(self, data_x, data_y, pulse_pars, pulse_lows, pulse_highs):
        
        # Define a function that conducts a local search for the amplitudes
        # that minimize the SSD.
        def locSearch(f):
            
            # Do some stuff
            
            # Return the minimizing parameters
            #return opt_pars
            return 0
        # Define the function to be minimized.
        
        SSD = np.zeros(np.shape(data_y))
        
        # Minimize sum((iPRC-data_y)**2)
        #          sum_i( 1/(amp_i-low_i)**2 + 1/(high_i-amp_i)**2 )
        # where iPRC = sum_i(amp_i*uPRC_i)
        
        return 0
        
#---------------------------------------------------------------------