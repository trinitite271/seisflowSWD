#!/usr/bin/env python3
"""
A seismic inversion (a.k.a full waveform inversion, adjoint tomography, full
waveform tomography) perturbs seismic velocity models by minimizing objective
functions defining differences between observed and synthetic waveforms.

This seismic inversion workflow performs a linear set of tasks involving:

1) Generating synthetic seismograms using an external numerical solver
2) Calculating time-dependent misfit (adjoint sources) between data
    (or other synthetics) and synthetics
3) Using adjoint sources to generate misfit kernels defining volumetric
    perturbations sensitive to data-synthetic misfit
4) Smoothing and summing misfit kernels into a single gradient
5) Perturbing the starting model with the gradient to reduce misfit defined by
    the objective function during a line search

The Inversion workflow runs the above tasks in a loop (iterations) while
exporting updated models, kernels and/or gradients to disk.
"""
import os
import sys
import numpy as np
from glob import glob
from seisflows import logger
from seisflows.workflow.inversion import Inversion
from seisflows.tools import msg, unix
from seisflows.tools.model import Model
from seisflows.tools.config import get_task_id
import matplotlib.pyplot as plt
import pathlib
import shutil
from obspy import Stream
from seisflows.solver.specfem2dWd import norm_trace
from scipy.special import softmax
from seisflows.preprocess.default import read, write, rename_as_adjoint_source

class WdInversion(Inversion):
    """
    WD Workflow
    ------------------
    Peforms iterative nonlinear inversion using the machinery of the Forward
    and Migration workflows, as well as a built-in optimization library.

    Parameters
    ----------
    :type start: int
    :param start: start inversion workflow at this iteration. 1 <= start <= inf
    :type end: int
    :param end: end inversion workflow at this iteration. start <= end <= inf
    :type iteration: int
    :param iteration: The current iteration of the workflow. If NoneType, takes
        the value of `start` (i.e., first iteration of the workflow). User can
        also set between `start` and `end` to resume a failed workflow.
    :type thrifty: bool
    :param thrifty: a thrifty inversion skips the costly intialization step
        (i.e., forward simulations and misfit quantification) if the final
        forward simulations from the previous iterations line search can be
        used in the current one. Requires L-BFGS optimization.
    :type export_model: bool
    :param export_model: export best-fitting model from the line search to disk.
        If False, new models can be discarded from scratch at any time.

    Paths
    -----
    :type path_eval_func: str
    :param path_eval_func: scratch path to store files for line search objective
        function evaluations, including models, misfit and residuals
    ***
    """
    __doc__ = Inversion.__doc__ + __doc__


    def __init__(self, vpmax, vpmin, vsmax, vsmin, vmax, vmin, fmax, fmin ,df ,dt ,ng ,ns, dg ,M, w, **kwargs):
        """
        Same as Inversion
        """
        self.vpmax = vpmax
        self.vpmin = vpmin
        self.vsmax = vsmax
        self.vsmin = vsmin
        self.vmax = vmax
        self.vmin = vmin
        self.fmax = fmax
        self.fmin = fmin
        self.df = df
        self.dt = dt
        self.ng = ng
        self.ns = ns       
        self.dg = dg
        self.M = M
        self.w = w
        self.np1 = vmax-vmin+1
        self.npair = int(np.round((fmax-fmin)/df))
        space_M = np.zeros((self.np1,self.npair))
        for i in range(self.npair):
            space_M[:,i]=np.arange(1,self.np1+1)
        self.space_M = space_M
        np.seterr(all='ignore')
        print('initWD')
        super().__init__(**kwargs)
        self.path['dispersion'] = os.path.join(self.path.workdir, "dispersion")


    @property
    def evaluation(self):
        """
        Convenience string return for log messages that gives the iteration
        and step count of the current evaluation as a formatted string
        e.g., i01s00
        """
        return f"i{self.iteration:0>2}s{self.optimize.step_count:0>2}"


    @property
    def task_list(self):
        """
        USER-DEFINED TASK LIST. This property defines a list of class methods
        that take NO INPUT and have NO RETURN STATEMENTS. This defines your
        linear workflow, i.e., these tasks are to be run in order from start to
        finish to complete a workflow.

        This excludes 'check' (which is run during 'import_seisflows') and
        'setup' which should be run separately

        .. note::
            For workflows that require an iterative approach (e.g. inversion),
            this task list will be looped over, so ensure that any setup and
            teardown tasks (run once per workflow, not once per iteration) are
            not included.

        :rtype: list
        :return: list of methods to call in order during a workflow
        """
        return [self.generate_synthetic_data,
                self.evaluate_initial_misfit,
                self.run_adjoint_simulations,
                self.postprocess_event_kernels,
                self.evaluate_gradient_from_kernels,
                self.initialize_line_search,
                self.evaluate_line_search_misfit,
                self.update_line_search,
                self.finalize_iteration
                ]


    @property
    def is_thrifty(self):
        """
        Thrifty inversions are a special case of inversion where the forward
        simulations and misfit quantification from the previous iteration's
        line search can be re-used as the forward simulation of the current iter

        This status check determines whether a thrifty iteration can be 
        performed, which is dependent on where we are in the inversion, and
        whether the optimization module has been restarted.

        .. warning::

            Thrifty status from previous iteration is NOT saved, so if your 
            workflow fails at `evaluate_initial_misfit`, the boolean check
            will fail and the workflow will re-evaluate the initial misfit.

        :rtype: bool
        :return: thrifty status, True if we can re-use previous forward sims
            False if we must go the normal inversion route
        """
        if self.thrifty is False:
            _thrifty_status = False
        elif self.iteration == self.start:
            logger.info("thrifty inversion encountering first iteration, "
                        "defaulting to standard inversion workflow")
            _thrifty_status = False
        elif self.optimize._restarted:  # NOQA
            logger.info("optimization has been restarted, defaulting to "
                        "standard inversion workflow")
            _thrifty_status = False
        elif self.iteration == self.end:
            logger.info("thrifty inversion encountering final iteration, "
                        "defaulting to inversion workflow")
            _thrifty_status = False
        else:
            logger.info("acceptable conditions for thrifty inverison, "
                        "continuing with thrifty inversion")
            _thrifty_status = True

        return _thrifty_status


    def check(self):
        """
        Checks inversion-specific parameters
        """
        super().check()


    def setup(self):
        """
        Assigns modules as attributes of the workflow. I.e., `self.solver` to
        access the solver module (or `workflow.solver` from outside class)

        Lays groundwork for inversion by running setup() functions for the
        involved sub-modules, generating True model synthetic data if necessary,
        and generating the pre-requisite database files.
        """
        super().setup()
        # _required_structure = self.solver.source_names
        # source_paths = [self.path.dispersion]
        _required_structure = {'Placeholders'}
        source_paths = [os.path.join(self.path.dispersion, source_name)
                for source_name in self.solver.source_names]
        source_paths = [p for p in source_paths if not os.path.exists(p)]
        self.solver.initialize_disp_directory(_required_structure, source_paths)


    def run(self):
        """Call the forward.run() function iteratively, from `start` to `end`
        Call the Task List in order to 'run' the workflow. Contains logic for
        to keep track of completed tasks and avoids re-running tasks that have
        previously been completed (e.g., if you are restarting your workflow)
        """
        self.offset = 48
        for self.iteration in  range(self.start,self.end + 1):
            logger.info(msg.mjr(f"RUNNING ITERATION {self.iteration:0>2}"))
            self.first_count=0
            n = 0  # To keep track of number of tasks completed
            for func in self.task_list:
                # Skip over functions which have already been completed
                if (func.__name__ in self._states.keys()) and (
                        self._states[func.__name__] == 1):  # completed
                    logger.info(f"'{func.__name__}' has already been run, skipping")
                    continue
                # Otherwise attempt to run functions that have failed or are
                # encountered for the first time
                else:
                    try:
                        # Print called func name, e.g., test_func -> TEST FUNC
                        _log_name = func.__name__.replace("_", " ").upper()
                        logger.info(msg.mnr(_log_name))
                        func()
                        n += 1
                        self._states[func.__name__] = 1  # completed
                        self.checkpoint()
                    except Exception as e:
                        self._states[func.__name__] = -1  # failed
                        self.checkpoint()
                        raise
                # Allow user to prematurely stop a workflow after a given task
                if self.stop_after and func.__name__ == self.stop_after:
                    logger.info(
                        msg.mjr(f"STOP WORKFLOW (`stop_after`={self.stop_after})")
                        )
                    break

            self.checkpoint()
            logger.info(f"completed {n} tasks from task list")

            if self.stop_after is None:
                logger.info(msg.mjr(f"COMPLETE ITERATION {self.iteration:0>2}"))
                self.iteration += 1
                logger.info(f"setting current iteration to: {self.iteration}")
                # Set the state file to pending for new iteration
                self._states = {key: 0 for key in self._states}
                self.checkpoint()
            else:
                break

            if self.optimize._line_search.loop_states == "FAIL":
                self.offset = self.offset - 3
                print(self.offset)
                logger.info(f"#######################current offset: {self.offset}")
                if self.offset == 3:
                    break
        
        evaluate_inv_result()


    def checkpoint(self):
        """
        Add an additional line in the state file to keep track of iteration
        """
        super().checkpoint()


    def generate_synthetic_data(self, **kwargs):
        """
        Generating dispersion data based on Inversion class
        """
        
        super().generate_synthetic_data(**kwargs)

        # read_data_shot()
        # i_shot = get_task_id()

        # path_shot = os.path.join(self.path.data, self.solver.source_name)
        # for offset in range(48,3,-3):
        #     self.offset = offset
        # self.generate_dispersion_curve(**kwargs)

        if self.iteration==1 and self.generate_data:
            for offset in range(48,3,-3):
                self.offset = offset
                self.generate_dispersion_curve(**kwargs)
            self.offset = 48


    def evaluate_objective_function(self, save_residuals=False, components=None,
                                    **kwargs):
        """
        Uses the preprocess module to evaluate the misfit/objective function
        given synthetics generated during forward simulations

        .. note::

            Must be run by system.run() so that solvers are assigned individual
            task ids/ working directories.

        :type save_residuals: str
        :param save_residuals: if not None, path to write misfit/residuls to
        :type components: list
        :param components: optional list of components to ignore preprocessing
            traces that do not have matching components. The adjoint sources for
            these components will be 0. E.g., ['Z', 'N']. If None, all available
            components will be considered.
        """
        # These are only required for overriding workflows which may hijack
        # this function to provide specific arguments to preprocess module
        iteration = kwargs.get("iteration", 1)
        step_count = kwargs.get("step_count", 0)
        save_adjsrcs = kwargs.get("save_adjsrcs", 
                                  os.path.join(self.solver.cwd, "traces", "adj")
                                  )

        if self.preprocess is None:
            logger.debug("no preprocessing module selected, will not evaluate "
                         "objective function")
            return

        if save_residuals:
            # Check that the calling workflow has properly set the string fmtr.
            assert ("{src}" in save_residuals), (
                "objective function evaluation requires string formatter {} " 
                f"in `save_residuals`: {save_residuals}"
            )
            save_residuals = save_residuals.format(src=self.solver.source_name)

        if self.export_residuals:
            export_residuals = os.path.join(self.path.output, "residuals")
        else:
            export_residuals = False

        logger.debug(f"quantifying misfit with "
                     f"'{self.preprocess.__class__.__name__}'")
        
        self.quantify_misfit(
            source_name=self.solver.source_name, components=components,
            save_adjsrcs=save_adjsrcs, save_residuals=save_residuals,
            export_residuals=export_residuals,
            iteration=iteration, step_count=step_count
        )


    def sum_residuals(self, residuals_files, save_to):
        """
        Convenience function to read in text files containing misfit residual
        information written by `preprocess.quantify_misfit` for each event, and
        sum the total misfit for the evaluation in a given optimization vector.

        Follows Tape et al. 2010 equations 6 and 7

        :type residuals_files: list of str
        :param residuals_files: pathnames to residuals files for each source,
            generated by the preprocessing module. Will be read in and summed
            to provide total misfit
        :type save_to: str
        :param save_to: name of Optimization module vector to save the misfit 
            value 'f', options are 'f_new' for misfit of current accepted model
            'm_new', or 'f_try' for the misfit of the current line search trial
            model
        :rtype: float
        :return: sum of squares of residuals, total misfit
        """
        # Catch empty files because usually we feed this function with a glob
        if not residuals_files:
            logger.critical(
                msg.cli(f"Residuals files not found for {self.evaluation}, "
                        f"preprocessing may have failed. Please check logs.",
                        border="=", header="preprocessing failed")
                )
            sys.exit(-1)

        event_misfits = []
        for residuals_file in residuals_files:
            event_misfit = np.loadtxt(residuals_file)
            # Some preprocessing modules only return a single misfit value
            # which will fail when called with len()
            try:
                num_measurements = len(event_misfit)
            except TypeError:
                num_measurements = 1
            # Tape et al. (2010) Equation 6
            event_misfit = np.sum(event_misfit) / (2. * num_measurements)
            event_misfits.append(event_misfit)

        # Tape et al. (2010) Equation 7
        total_misfit = np.sum(event_misfits) / len(event_misfits)

        # Sum newly calc'd residuals into the optimization library
        self.optimize.save_vector(name=save_to, m=total_misfit)
        logger.info(f"misfit {save_to} ({self.evaluation}) = "
                    f"{total_misfit:.3E}")


    def evaluate_initial_misfit(self, **kwargs):
        
        super().evaluate_initial_misfit(**kwargs)


    def prepare_data_for_solver(self, _src=None, _copy_function=unix.ln,
                                **kwargs):
        """

        """
        # super().prepare_data_for_solver(**kwargs)
        # Location to store 'observation' data
        dst = os.path.join(self.solver.cwd, "disp", "obs", "")

        # Check if there is data already in the directory, User may have
        # manually input it here, or we are on iteration > 1 so data has already
        # been prepared, either way, make sure we don't overwrite it
        if glob(os.path.join(dst, "*")):
            logger.warning(f"data already found in "
                           f"{self.solver.source_name}/disp/obs/*, "
                           f"skipping data preparation"
                           )
            return

        logger.info(f"preparing observation dispersion curve for source "
                    f"{self.solver.source_name}")
        src = _src or os.path.join(self.path.dispersion,
                                   self.solver.source_name, "*")
        logger.debug(f"looking for data in: '{src}'")

        # If no data are found, exit this process, as we cannot continue
        if not glob(src):
            logger.critical(msg.cli(
                f"{self.solver.source_name} found no `obs` data with "
                f"wildcard: '{src}'. Please check `path_data` or manually "
                f"import data and re-submit", border="=",
                header="data import error")
            )
            sys.exit(-1)

        for src_ in glob(src):
            # Symlink or copy data to scratch dir. (symlink by default)
            _copy_function(src_, dst)


    def run_forward_simulations(self, **kwargs):
        """
        Same as Inversion
        """

        super().run_forward_simulations(**kwargs)


    def _run_adjoint_simulation_single(self, save_kernels=None,
                                       export_kernels=None, **kwargs):
        """
        Overrides 'workflow.migration._run_adjoint_simulation_single' to hijack
        the default path location for exporting kernels to disk
        """
        # Set default value for `export_kernels` or take program default

        super()._run_adjoint_simulation_single(save_kernels=save_kernels,
                                               export_kernels=export_kernels,
                                               **kwargs)
        save_kernels = os.path.join(self.path.eval_grad, "kernels",
                                        self.solver.source_name, "")
        for par in ['Hessian1','Hessian2']:
            kernel_files = glob(self.solver.model_wildcard(par=par, kernel=True))
            if kernel_files:
                unix.mv(src=kernel_files, dst=save_kernels)


    def evaluate_gradient_from_kernels(self):
        """
        Overwrite `workflow.migration` to convert the current model and the
        gradient calculated by migration from their native SPECFEM model format
        into optimization vectors that can be used for model updates.

        Also includes search direction computation, which takes the gradient
        `g_new` and scales to provide an appropriate search direction. At 
        the simplest form (gradient descent), the search direction is simply -g
        """
        super().evaluate_gradient_from_kernels()


    def initialize_line_search(self):
        """
        Computes search direction using the optimization library and sets up
        line search machinery to 'perform line search' by placing correct files
        on disk for each of the modules to find.

        Optimization module perturbs the current model (m_new) by the search
        direction (p_new) to recover the trial model (m_try). This model is
        then exposed on disk to the solver.
        """
        # Set up the line search machinery. Step count forced to 1
        self.optimize.initialize_search()

        # Determine the model we will use for first line search step count
        # m_try = m_new + alpha * p_new (i.e., m_i+1 = m_i + dm)
        alpha, _ = self.optimize.calculate_step_length()
        m_try = self.optimize.compute_trial_model(alpha=alpha)
        m_try = self.update_model_bound(m_try)

        # Save the current state of the optimization module to disk
        self.optimize.save_vector(name="m_try", m=m_try)
        self.optimize.save_vector(name="alpha", m=alpha)
        self.optimize.checkpoint()

        # Expose model `m_try` to the solvers by placing it in eval_func dir.
        _path_m_try = os.path.join(self.path.eval_func, "model")
        m_try.write(path=_path_m_try)


    def evaluate_line_search_misfit(self):
        """
        Evaluate line search misfit `f_try` by running forward simulations 
        through the trial model `m_try` and comparing with observations.
        Acts like a stripped down version of `evaluate_initial_misfit`

        TODO Add in export traces functionality, need to honor step count
        """
        logger.info(msg.sub(f"LINE SEARCH STEP COUNT "
                            f"{self.optimize.step_count:0>2}"))
        
        logger.info(f"`m_try` model parameters for line search evaluation:")
        self.solver.check_model_values(path=os.path.join(self.path.eval_func, 
                                                         "model"))

        self.system.run(
            [self.run_forward_simulations,
             self.evaluate_objective_function],
            path_model=os.path.join(self.path.eval_func, "model"),
            save_residuals=os.path.join(
                self.path.eval_func, "residuals",
                f"residuals_{{src}}_{self.evaluation}.txt")
        )

        residuals_files = glob(os.path.join(
            self.path.eval_func, "residuals", 
            f"residuals_*_{self.evaluation}.txt")
            )
        assert residuals_files, (
                f"No residuals files found for evaluation {self.evaluation} "
                f"Please check preprocessing log files."
                )
        # Read in all avail residual files generated by preprocessing module
        self.sum_residuals(residuals_files, save_to="f_try")
    

    def update_line_search(self):
        """
        Given the misfit `f_try` calculated in `evaluate_line_search_misfit`,
        use the Optimization module to determine if the line search has passed,
        failed, or needs to perform a subsequent step.

        The line search state machine acts in the following way:
        - Pass: Run clean up and proceed with workflow
        - Try: Re-calculate step length (alpha) and re-evaluate misfit (f_try)
        - Fail: Try to restart optimization module and restart line search. If
            still failing, exit workflow

        .. note::

            Line search starts on step_count == 1 because step_count == 0 is
            considered the misfit of the starting model
        """
        # Update line search history with the step length (alpha) and misfit (f)
        # and incremement the step count
        self.optimize.update_search()
        alpha, status = self.optimize.calculate_step_length()
        m_try = self.optimize.compute_trial_model(alpha=alpha)
        m_try = self.update_model_bound(m_try)
        
        # Save new model (m_try) and step length (alpha) for new trial step
        if alpha is not None:
            self.optimize.save_vector("alpha", alpha)
        if m_try is not None:
            self.optimize.save_vector("m_try", m_try)

        # Proceed based on the outcome of the line search
        if status.upper() == "PASS":
            # Save outcome of line search to disk; reset step to 0 for next iter
            logger.info("trial step successful. finalizing line search")

            # Finalizing line search sets `m_try` -> `m_new` for later iters
            self.optimize.finalize_search()
            self.optimize.checkpoint()
            return
        elif status.upper() == "TRY":
            logger.info("trial step unsuccessful. re-attempting line search")

            # Expose the new model to the solver directories for the next step
            _path_m_try = os.path.join(self.path.eval_func, "model")
            m_try.write(path=_path_m_try)

            # Re-set state file to ensure that job failure will recover
            self._states["evaluate_line_search_misfit"] = 0

            # Recursively run the line search to get a new misfit
            self.optimize.checkpoint()
            self.evaluate_line_search_misfit()
            self.update_line_search()  # RECURSIVE CALL
            self.loop_states = status.upper()
        elif status.upper() == "FAIL":
            self.loop_states = status.upper()
            # Check if we are able to restart line search w/ new parameters
            # if self.optimize.attempt_line_search_restart():
            #     logger.info("line search has failed. restarting "
            #                 "optimization algorithm and line search.")
            #     # Reset the line search machinery; set step count to 0
            #     self.optimize.restart()

            #     # Re-set state file to ensure that job failure will recover
            #     self._states["evaluate_line_search_misfit"] = 0

            #     # Restart the entire line search procedure
            #     self.optimize.checkpoint()
            #     self.initialize_line_search()
            #     self.evaluate_line_search_misfit()
            #     self.update_line_search()  # RECURSIVE CALL
            # # If we can't then line search has failed. Abort workflow
            # else:
            #     logger.critical(
            #         msg.cli("Line search has failed to reduce the misfit and "
            #                 "has run out of fallback options. Aborting "
            #                 "inversion.", border="=",
            #                 header="line search failed")
            #     )
            #     self.loop_states = status.upper()
                # sys.exit(-1)


    def finalize_iteration(self):
        """
        WD require force update model to its max and min value.
        """
        m_new = self.optimize.load_vector("m_new")
        m_new = self.update_model_bound(m_new)
        self.optimize.save_vector("m_new", m_new)
        logger.info(f"Force update of model max and min bounds")

        super().finalize_iteration()


    def update_model_bound(self,model):
        """
        Update model bound
        """
        if model is None:
            return model
        for parameter in model.parameters:
            if parameter == "vp":
                v_max = self.vpmax
                v_min = self.vpmin
            elif parameter == "vs":
                v_max = self.vsmax
                v_min = self.vsmin
            model.model[parameter] = np.where(model.model[parameter] > v_max, v_max, model.model[parameter])
            model.model[parameter] = np.where(model.model[parameter] < v_min, v_min, model.model[parameter])
        model.model['vp'] = 1.732 * model.model['vs']
        model.merge()
        logger.debug(f"Force update of model max and min bounds")
        return model
    
    
    def generate_dispersion_curve(self, **kwargs):
        """
        For synthetic inversion cases, we can use the workflow machinery to
        generate 'data' by running simulations through a target/true model for 
        each of our `ntask` sources. This only needs to be run once during a 
        workflow.
        """

        # Check the target model that will be used to generate data
        # logger.info("checking true/target model parameters:")
        # self.solver.check_model_values(path=self.path.model_true)

        self.system.run([self._generate_dispersion_curve_single], **kwargs)


    def _generate_dispersion_curve_single(self,_copy_function=unix.ln,**kwargs):


        save_disp = os.path.join(self.path.dispersion, self.solver.source_name,'offset'+str(self.offset))
        path_shot = os.path.join(self.path.data, self.solver.source_name)
        # Run forward simulation with solver
        i_shot = get_task_id() 
        self.run_dispersion_simulations(i_shot, path_shot,save_disp=save_disp)
        
        # Symlink data into solver directories so it can be found by preprocess
        src = os.path.join(save_disp, "*")
        dst = os.path.join(self.solver.cwd, "disp", "obs")

        for src_ in glob(src):
            _copy_function(src_, dst)    


    def run_dispersion_simulations(self,i_shot, path_shot, save_disp=None, **kwargs):
        
        seismo_v = self.solver.read_shot(path_shot)
        # print(i_shot)
        # print('1')
        # print(np.argmax(seismo_v))
        # print(path_shot)
        # print(self.df,self.dt,self.vmin,self.vmax,self.fmin,self.fmax,self.ng,self.dg,self.offset,self.M)
        if save_disp is None:
            # save_disp = os.path.join(self.solver.cwd, "disp", "syn",'offset'+str(self.offset)+'s'+self.solver.source_name)
            save_disp = os.path.join(self.solver.cwd, "disp", "syn",'offset'+str(self.offset))
        mlr = 0
        mll = 0
        cr_pre_r = np.array([0])
        cr_pre_l = np.array([0])
        if i_shot<=self.ns-np.round(self.w/self.M):
            mlr,_ = self.solver.RTpr(seismo_v,i_shot,self.df,self.dt,self.vmin,self.vmax,self.fmin,self.fmax,self.ng,self.dg,self.offset,self.M)
            cr_pre_r = self.solver.extractDispCurve(mlr)
        save_disp1 = save_disp + 'r'
        np.save(save_disp1,cr_pre_r)

        if i_shot>=np.round(self.w/self.M)+1:
            mll,_ = self.solver.RTpl(seismo_v,i_shot,self.df,self.dt,self.vmin,self.vmax,self.fmin,self.fmax,self.ng,self.dg,self.offset,self.M)
            cr_pre_l = self.solver.extractDispCurve(mll)
        save_disp1 = save_disp + 'l'
        np.save(save_disp1,cr_pre_l)
        if i_shot==0:
            mlr = norm_trace(mlr)
            fig,ax = plt.subplots(1)
            plt.pcolor(mlr)
            plt.plot(cr_pre_r)
            fig.savefig(save_disp1+'ml.png')
            fig.clf()
            plt.close()
        # return mlr, mll, cr_pre_r, cr_pre_l


    def quantify_misfit(self, source_name=None, save_residuals=None,
                        export_residuals=None, save_adjsrcs=None,
                        components=None, **kwargs):
        """
 
        """
        # self._iteration = iteration
        # self._step_count = step_count

        # Retrieve matching obs and syn trace filenames to run through misfit
        # and initialize empty adjoint sources

        # Initialize empty adjoint sources for all synthetics that may or may
        # not be overwritten by the misfit quantification step
        path_pre = os.path.join(self.solver.path.scratch, source_name, "disp", "syn",'offset'+str(self.offset))
        path_obs = os.path.join(self.solver.path.scratch, source_name, "disp", "obs",'offset'+str(self.offset))

        i_shot = get_task_id() 
        path_shot = os.path.join(self.solver.path.scratch, source_name, "traces", "syn")
        seismo_v = self.solver.read_shot(path_shot)
        [nt,ng] = np.shape(seismo_v)
        mlr = 0
        mll = 0
        cr_pre_r = np.array([0])
        cr_pre_l = np.array([0])
        saveForBackwardr = np.array([0])
        saveForBackwardl = np.array([0])
        if i_shot<=self.ns-np.round(self.w/self.M):
            mlr,saveForBackwardr = self.solver.RTpr(seismo_v,i_shot,self.df,self.dt,self.vmin,self.vmax,self.fmin,self.fmax,ng,self.dg,self.offset,self.M)
            cr_pre_r = self.solver.extractDispCurve(mlr)
        save_disp1 = path_pre + 'r'
        np.save(save_disp1,cr_pre_r)

        if i_shot>=np.round(self.w/self.M)+1:
            mll,saveForBackwardl = self.solver.RTpl(seismo_v,i_shot,self.df,self.dt,self.vmin,self.vmax,self.fmin,self.fmax,ng,self.dg,self.offset,self.M)
            cr_pre_l = self.solver.extractDispCurve(mll)
        save_disp1 = path_pre + 'l'
        np.save(save_disp1,cr_pre_l)


        cr_obs_r = np.load(path_obs+'r.npy')
        cr_obs_l = np.load(path_obs+'l.npy')
        grad_outputr = cr_pre_r - cr_obs_r
        grad_outputl = cr_pre_l - cr_obs_l
        space_M = self.space_M
        # residuals = np.concatenate((grad_outputr**2,grad_outputl**2)).tolist()
        residuals = (grad_outputr**2).tolist()
        # Write residuals to text file for other modules to find
        if save_residuals:
            with open(save_residuals, "a") as f:
                for residual in residuals:
                    f.write(f"{residual:.2E}\n")

            # Exporting residuals to disk (output/) for more permanent storage
            if export_residuals:
                unix.mkdir(export_residuals)
                unix.cp(src=save_residuals, dst=export_residuals)

        logger.info(f"FINISH QUANTIFY MISFIT: {source_name}")

        logger.info(f"#######################self.optimize.step_count: {self.optimize.step_count}")
        logger.info(f"#######################self.first_count: {self.first_count}")
        if self.first_count==0:
            self.first_count = self.first_count + 1
            a_data_res = self.SWDgrad_1(i_shot,space_M,nt,ng,grad_outputr,grad_outputl,saveForBackwardr,saveForBackwardl,SoftArgNorm=1e+4)

            path_data = pathlib.Path(os.path.join(self.solver.path.scratch, source_name, "traces", "syn"))
            syn_fid_all = sorted(path_data.glob("*BXZ.semd"))

            for i in range(ng):
                syn_fid = syn_fid_all[i]
                syn = read(fid=syn_fid, data_format=self.preprocess.syn_data_format)
                tr_syn = syn[0]
                if save_adjsrcs:
                    adjsrc = tr_syn.copy()
                    adjsrc.data = a_data_res[:,i]
                    fid = os.path.basename(syn_fid)
                    fid = rename_as_adjoint_source(fid, fmt=self.preprocess.syn_data_format)
                    write(st=Stream(adjsrc), fid=os.path.join(save_adjsrcs, fid),
                            data_format=self.preprocess.syn_data_format)
                else:
                    adjsrc = None

            syn_fid_all = sorted(path_data.glob("*BXX.semd"))

            for i in range(ng):
                syn_fid = syn_fid_all[i]
                syn = read(fid=syn_fid, data_format=self.preprocess.syn_data_format)
                tr_syn = syn[0]
                if save_adjsrcs:
                    adjsrc = tr_syn.copy()
                    adjsrc.data = a_data_res[:,i]*0.0
                    fid = os.path.basename(syn_fid)
                    fid = rename_as_adjoint_source(fid, fmt=self.preprocess.syn_data_format)
                    write(st=Stream(adjsrc), fid=os.path.join(save_adjsrcs, fid),
                            data_format=self.preprocess.syn_data_format)
                else:
                    adjsrc = None


    def SWDgrad_1(self,i_shot,space_M,nt,ng,grad_outputr,grad_outputl,saveForBackwardr,saveForBackwardl,SoftArgNorm=1e+4):
        a_data_res = np.zeros((nt,ng))
        zxcq1 = np.seterr(all='ignore')
        if i_shot<=self.ns-np.round(self.w/self.M):
            ll0 = saveForBackwardr['ll0']
            lf = saveForBackwardr['lf']
            nf = saveForBackwardr['nf']
            ccn = saveForBackwardr['ccn']
            uxtposr = saveForBackwardr['uxtposr']
            offsetr = np.size(uxtposr)
            mlr = saveForBackwardr['mlr']
            mlr = (mlr)**(1/40)
            mmr = saveForBackwardr['mm']
            ml_Nr = softmax(mlr*SoftArgNorm,axis=0)
            gradmlr = np.zeros_like(ml_Nr)
            # mmr = mmr[:,(lf-1):nf]
            abs_mmr = np.abs(mmr)
            for i in range(self.npair):
                test = ml_Nr[:,i]*space_M[:,i] - np.sum((space_M[:,i]*ml_Nr[:,i]) * ml_Nr[:,i].reshape(-1,1),axis=1)
                gradmlr[:,i] = test * 2 * (grad_outputr[i])*1
            gradmmr = mmr/np.abs(mmr)
            gradmmr = np.divide(mmr, abs_mmr, out=np.zeros_like(mmr), where=abs_mmr!=0)
            gradmmr = gradmmr * gradmlr
            graddr = np.zeros((offsetr,ccn))
            for luoj in range(lf-1,nf):
                    temps = np.dot(np.exp(ll0*(luoj)).conj().T, gradmmr[:,luoj-lf+1])
                    graddr[:,luoj] = temps
            
            graduxtr = np.real(np.fft.ifft(graddr,ccn,1))
            graduxtr = graduxtr[:,:nt]
            a_data_res[:,uxtposr] = graduxtr.T

        if i_shot>=np.round(self.w/self.M)+1:
            ll0 = saveForBackwardl['ll0']
            lf = saveForBackwardl['lf']
            nf = saveForBackwardl['nf']
            ccn = saveForBackwardl['ccn']
            uxtposl = saveForBackwardl['uxtposl']
            offsetl = np.size(uxtposl)
            mll = saveForBackwardl['mll']
            mll = (mll)**(1/40)
            mml = saveForBackwardl['mm']
            ml_Nl = softmax(mll*SoftArgNorm,axis=0)
            gradmll = np.zeros_like(ml_Nl)
            # mml = mml[:,(lf-1):nf]
            abs_mml = np.abs(mml)
            for i in range(self.npair):
                test = ml_Nl[:,i]*space_M[:,i] - np.sum((space_M[:,i]*ml_Nl[:,i]) * ml_Nl[:,i].reshape(-1,1),axis=1)
                gradmll[:,i] = test * 2 * (grad_outputl[i])*1
            gradmml = mml/np.abs(mml)
            gradmml = np.divide(mml, abs_mml, out=np.zeros_like(mml), where=abs_mml!=0)
            gradmml = gradmml * gradmll
            graddl = np.zeros((offsetl,ccn))
            for luoj in range(lf-1,nf):
                    temps = np.dot(np.exp(ll0*(luoj)).conj().T, gradmml[:,luoj-lf+1])
                    graddl[:,luoj] = temps

            graduxtl = np.real(np.fft.ifft(graddl,ccn,1))
            graduxtl = graduxtl[:,:nt]
            a_data_res[:,uxtposl[:-1]] = graduxtl[:0:-1,:].T
        a_data_res = a_data_res/np.max(np.abs(a_data_res))
        return a_data_res
    

    def postprocess_event_kernels(self):
        """
        Combine/sum NTASK event kernels into a single volumetric kernel and
        then (optionally) smooth the output misfit kernel by convolving with
        a 3D Gaussian function with user-defined horizontal and vertical
        half-widths, or by using the smooth2a.
        .. note::

            If you hit a floating point error during the smooth operation, 
            your kernels may be zero due to something going awry in the
            misfit quantification or adjoint simulations.


            Difference from inversion class: adds preconditioned hessian matrix, 
            respectively smooth kernel, then computes gradient from hessian
        """

        def combine_event_kernels(**kwargs):
            """
            Combine individual event kernels into a misfit kernel for each
            parameter defined by the solver
            """
            # Input paths are the kernels generated by each of the sources
            input_paths = [os.path.join(self.path.eval_grad, "kernels", src) for
                           src in self.solver.source_names]

            logger.info("combining event kernels into single misfit kernel")

            # Parameters to combine are the kernels, which follow the 
            # naming convention {par}_kernel
            kernels = self.solver._parameters
            kernels.append('Hessian1')
            kernels.append('Hessian2')
            parameters = [f"{par}_kernel" for par in self.solver._parameters+['Hessian1', 'Hessian2']]

            self.solver.combine(
                input_paths=input_paths,
                output_path=os.path.join(self.path.eval_grad, "misfit_kernel"),
                parameters=parameters
            )

        def smooth_misfit_kernel(**kwargs):
            """
            Smooth the misfit kernel using the underlying Solver smooth function
            """
            if self.solver.smooth_h > 0. or self.solver.smooth_v > 0.:
                logger.info(
                    f"smoothing misfit kernel: "
                    f"H={self.solver.smooth_h}; V={self.solver.smooth_v}"
                )
                parameters = self.solver._parameters+['Hessian1', 'Hessian2']
                # Make a distinction that we have a pre- and post-smoothed kern.
                unix.mv(
                    src=os.path.join(self.path.eval_grad, "misfit_kernel"),
                    dst=os.path.join(self.path.eval_grad, "mk_nosmooth")
                )
                self.solver.smooth(
                    input_path=os.path.join(self.path.eval_grad, "mk_nosmooth"),
                    output_path=os.path.join(self.path.eval_grad,"misfit_kernel",),
                    parameters=parameters
                )

        def update_with_hessian(**kwargs):
            misfit_kernel_path = os.path.join(self.path.eval_grad, "misfit_kernel")
            gradient = Model(path=misfit_kernel_path,parameters=['vp_kernel','vs_kernel','Hessian1_kernel', 'Hessian2_kernel'])
            for par in ['Hessian1_kernel', 'Hessian2_kernel']:
                maxh = np.max(np.abs(gradient.model[par]))
                if maxh < 1.e-18:
                    gradient.model[par][:] = 1.0
                else:
                    gradient.model[par] = gradient.model[par]/maxh
                gradient.model[par] = np.where(gradient.model[par]>1e-3,1./gradient.model[par],1./1e-3)
                # gradient.model[par] = 1./gradient.model[par]
                # gradient.model[par] = np.where(gradient.model[par]<10,gradient.model[par],10)
                # gradient.model[par] = np.where(gradient.model[par]>-10,gradient.model[par],-10)
                


            gradient.model['vp_kernel'] = gradient.model['vp_kernel']*gradient.model['Hessian1_kernel']
            gradient.model['vs_kernel'] = gradient.model['vs_kernel']*gradient.model['Hessian1_kernel']
            # gradient.model['vp_kernel'] = gradient.model['vp_kernel']*1
            # gradient.model['vs_kernel'] = gradient.model['vs_kernel']*1
            gradient.merge()
            gradient.write(path=misfit_kernel_path)

        # Make sure were in a clean scratch eval_grad directory
        tags = ["misfit_kernel", "mk_nosmooth"]
        for tag in tags:
            scratch_path = os.path.join(self.path.eval_grad, tag)
            if os.path.exists(scratch_path):
                shutil.rmtree(scratch_path)

        # NOTE: May need to increase the tasktime by a factor of n because the
        # smoothing operation is computational expensive; add the following:
        # tasktime=self.system.tasktime * 2  # increase 2 if you need more time
        self.system.run([combine_event_kernels, smooth_misfit_kernel,
                         update_with_hessian], single=True,
                         tasktime=self.system.tasktime * 2)
        

def evaluate_inv_result(workdir,vsName,**kwargs):
    path_eval = os.path.join(workdir, 'evaluation')
    if os.path.exists(path_eval):
        unix.rm(path_eval)
    unix.mkdir(path_eval)
    path_eval_model = os.path.join(workdir, 'output', vsName)
    path_eval_waveform = os.path.join(path_eval, 'waveform')
    path_eval_dispersion = os.path.join(path_eval, 'dispersion')
    run_list = [self.run_forward_simulations]


    self.system.run(run_list, path_model=path_eval_model, 
                    save_residuals=path_eval_waveform, 
                    save_forward_arrays=path_eval_waveform,
                    flag_save_forward=True,
                    **kwargs,
                    )
    
    for offset in range(48,3,-3):
        self.offset = offset
        self.generate_dispersion_curve(**kwargs)
        self.offset = 48
