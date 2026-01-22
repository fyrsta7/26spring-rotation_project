	return 0;
}

static void rttm_setup_timer(void __iomem *base)
{
	RTTM_DEBUG(base);
	rttm_stop_timer(base);
	rttm_set_period(base, 0);
}

static u64 rttm_read_clocksource(struct clocksource *cs)
{
	struct rttm_cs *rcs = container_of(cs, struct rttm_cs, cs);

	return (u64)rttm_get_counter(rcs->to.of_base.base);
}

/*
 * Module initialization part.
 */

static DEFINE_PER_CPU(struct timer_of, rttm_to) = {
	.flags				= TIMER_OF_BASE | TIMER_OF_CLOCK | TIMER_OF_IRQ,
	.of_irq = {
		.flags			= IRQF_PERCPU | IRQF_TIMER,
		.handler		= rttm_timer_interrupt,
	},
	.clkevt = {
		.rating			= 400,
		.features		= CLOCK_EVT_FEAT_PERIODIC | CLOCK_EVT_FEAT_ONESHOT,
		.set_state_periodic	= rttm_state_periodic,
		.set_state_shutdown	= rttm_state_shutdown,
		.set_state_oneshot	= rttm_state_oneshot,
		.set_next_event		= rttm_next_event
	},
};

static int rttm_enable_clocksource(struct clocksource *cs)
{
	struct rttm_cs *rcs = container_of(cs, struct rttm_cs, cs);

	rttm_disable_irq(rcs->to.of_base.base);
	rttm_setup_timer(rcs->to.of_base.base);
	rttm_enable_timer(rcs->to.of_base.base, RTTM_CTRL_TIMER,
			  rcs->to.of_clk.rate / RTTM_TICKS_PER_SEC);

	return 0;
}

struct rttm_cs rttm_cs = {
	.to = {
		.flags	= TIMER_OF_BASE | TIMER_OF_CLOCK,
	},
	.cs = {
		.name	= "realtek_otto_timer",
		.rating	= 400,
		.mask	= CLOCKSOURCE_MASK(RTTM_BIT_COUNT),
		.flags	= CLOCK_SOURCE_IS_CONTINUOUS,
		.read	= rttm_read_clocksource,
		.enable	= rttm_enable_clocksource
	}
};

static u64 notrace rttm_read_clock(void)
{
	return (u64)rttm_get_counter(rttm_cs.to.of_base.base);
}

static int rttm_cpu_starting(unsigned int cpu)
{
	struct timer_of *to = per_cpu_ptr(&rttm_to, cpu);

	RTTM_DEBUG(to->of_base.base);
	to->clkevt.cpumask = cpumask_of(cpu);
	irq_set_affinity(to->of_irq.irq, to->clkevt.cpumask);
	clockevents_config_and_register(&to->clkevt, RTTM_TICKS_PER_SEC,
					RTTM_MIN_DELTA, RTTM_MAX_DELTA);
	rttm_enable_irq(to->of_base.base);

	return 0;
}

static int __init rttm_probe(struct device_node *np)
{
	int cpu, cpu_rollback;
	struct timer_of *to;
	int clkidx = num_possible_cpus();
/*
 * Use the first n timers as per CPU clock event generators
 */
	for_each_possible_cpu(cpu) {
		to = per_cpu_ptr(&rttm_to, cpu);
		to->of_irq.index = to->of_base.index = cpu;
		if (timer_of_init(np, to)) {
			pr_err("%s: setup of timer %d failed\n", __func__, cpu);
			goto rollback;
		}
		rttm_setup_timer(to->of_base.base);
	}
/*
 * Activate the n'th+1 timer as a stable CPU clocksource.
 */
