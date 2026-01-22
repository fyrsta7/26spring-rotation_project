void prio_queue_reverse(struct prio_queue *queue)
{
	int i, j;

	if (queue->compare != NULL)
		die("BUG: prio_queue_reverse() on non-LIFO queue");
	for (i = 0; i < (j = (queue->nr - 1) - i); i++)
		swap(queue, i, j);
}
