from core.assignment_models import Dual, Primal, Probit
from core.optimizers import GradientDescent, FISTA, LineSearch

def get_gd(config, graph, ref_data=None, ref_opt=None):
    dir_ = 'NGEVSUE'+'_GD'+'_m'+str(config.m_min)+'_s'+str(config.step_size)
    optimizer = GradientDescent(init_lr=config.step_size)
    sue_gd = Dual(graph,
                optimizer,
                beta=config.beta,
                threshold=config.threshold,
                m_min=config.m_min,
                m_max=config.m_max,
                ref_data=ref_data, ref_opt=ref_opt,
                print_step=config.print_step
                )
    return sue_gd, dir_

def get_agd_BT(config, graph, ref_data=None, ref_opt=None):
    config.with_BT = True
    step = 1e-4
    dir_ = 'NGEVSUE'+'_AGD'+'_BT'+str(config.with_BT)+'_m'+str(config.m_min)+'_s'+str(step)+'_kmin'+str(config.k_min)+'_eta'+str(config.eta)
    optimizer = FISTA(init_lr=step,
                      k_min=config.k_min,
                      with_BT=config.with_BT,
                      eta=config.eta,
                      min_lr=config.min_s)
    sue_agd = Dual(graph,
                optimizer,
                beta=config.beta,
                threshold=config.threshold,
                m_min=config.m_min,
                m_max=config.m_max,
                ref_data=ref_data, ref_opt=ref_opt,
                print_step=config.print_step
                )
    return sue_agd, dir_

def get_agd(config, graph, ref_data=None, ref_opt=None):
    config.with_BT = False
    dir_ = 'NGEVSUE'+'_AGD'+'_BT'+str(config.with_BT)+'_m'+str(config.m_min)+'_s'+str(config.step_size)+'_kmin'+str(config.k_min)+'_eta'+str(config.eta)
    optimizer = FISTA(init_lr=config.step_size,
                      k_min=config.k_min,
                      with_BT=config.with_BT,
                      eta=config.eta)
    sue_agd = Dual(graph,
                optimizer,
                beta=config.beta,
                threshold=config.threshold,
                m_min=config.m_min,
                m_max=config.m_max,
                ref_data=ref_data, ref_opt=ref_opt,
                print_step=config.print_step
                )
    return sue_agd, dir_

def get_pl(config, graph, ref_data=None, ref_opt=None):
    m_min = config.m_min #min(config.m_min // 2, 500)
    m_max = config.m_max #min(config.m_max // 2, 500)
    dir_ = 'NGEVSUE'+'_'+config.line_search+'_m'+str(m_min)
    sue_pl = Primal(graph,
            line_search=LineSearch(
                method=config.line_search, threshold=config.ls_tolerance),
            beta=config.beta,
            threshold=config.threshold,
            m_min=m_min,
            m_max=m_max,
            ref_data=ref_data, ref_opt=ref_opt,
            print_step=config.print_step
            )
    return sue_pl, dir_

def get_msa(config, graph, ref_data=None, ref_opt=None):
    m_min = config.m_min
    m_max = config.m_max
    dir_ = 'NGEVSUE'+'_MSA'+'_m'+str(m_min)
    sue_msa = Primal(graph,
            line_search=LineSearch(
                method='MSA', threshold=config.ls_tolerance),
            beta=config.beta,
            threshold=config.threshold,
            m_min=m_min,
            m_max=m_max,
            ref_data=ref_data, ref_opt=ref_opt,
            print_step=config.print_step
            )
    return sue_msa, dir_

def get_probit(config, graph, ref_data=None, ref_opt=None):
    m_min = config.m_min
    m_max = config.m_max
    dir_ = 'ProbitSUE'+'_MSA'+'_m'+str(m_min)
    sue_probit = Probit(graph,
            line_search=LineSearch(
                method='MSA', threshold=config.ls_tolerance),
            beta=config.beta,
            threshold=config.threshold,
            m_min=m_min,
            m_max=m_max,
            ref_data=ref_data, ref_opt=ref_opt,
            print_step=config.print_step
            )
    return sue_probit, dir_
