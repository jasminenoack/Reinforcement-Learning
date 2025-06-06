# Tic Tac Logic

This is a puzzle game where you need to place an X or O in every square so that every row or column has the same number of Xs and Os and there are never more than 2 of any symbol next to each other orthogonally. Additionally, you want to ensure that no two rows or columns are identical.


# RLShapingBasedAgent

This agent isn't currently working super well. It sometimes does very well and sometimes quite poorly. It's primarily dependent on really detailed shaping data to ensure that it doesn't make poor life decisions. That shaping data is incomplete. So it finds some information it's extremely confident in, but then tends to be a bit lost.

I've been testing a lot of information around shaping and scoring to try to determine what's ideal here.

Part of the problem is that a lot of the information is relatively uncertain, which leads to that information actually being misleading at least some of the time.

It's not fully clear the best way to actually recover from this

# FailureAgent

This is an agent inspired by a coworker who once told me "I don't know the solution but I know what failure looks like". This agent is sort of randomly flopping around trying to find all of the pathways that will fail. This hopefully allows it to eventually find a success... not so far but some day.

# Mask Agent

This friend is in development. The idea here is to see if the agent can actually learn what might be useful. So basically we teach it how to look at the board and ask it to find patterns that it thinks are helpful. I think there could be something interesting here in that it might find patterns that do not appear to us.

The first step here is to attempt 3 masks that are related to 3 cell sequences with an empty cell in the center. The idea is to figure out if it is able to actually tell us what seems quite bad.

What we currently have working and semi tested (basically testing the parts with less confidence or visibility). We can create masks and then when we create the masks we train on them for a while. Once we have trained we start to make decisions based on what we have learned about the masks. If we learn trying to place an X between 2 Xs failed 30 times we avoid doing it any longer.

This is the first agent I've been able to get to actually win.

```
Results for env 4:
Average Actions per Episode: 24.34
Average Score per Episode: 141.07
Error Rate: 0.00%
Win Rate: 71.00%
----------------------------------------

Results for env 5:
Average Actions per Episode: 19.58
Average Score per Episode: 110.57
Error Rate: 0.00%
Win Rate: 0.00%
----------------------------------------

Results for env 6:
Average Actions per Episode: 22.57
Average Score per Episode: 96.50
Error Rate: 0.00%
Win Rate: 0.00%
----------------------------------------

Results for env 7:
Average Actions per Episode: 23.73
Average Score per Episode: 121.71
Error Rate: 0.00%
Win Rate: 66.00%
----------------------------------------

Results for env 8:
Average Actions per Episode: 24.49
Average Score per Episode: 95.60
Error Rate: 0.00%
Win Rate: 1.00%
----------------------------------------

Results for env 9:
Average Actions per Episode: 25.29
Average Score per Episode: 163.01
Error Rate: 0.00%
Win Rate: 78.00%
----------------------------------------
```

It's not always winning but it's very good at some boards. This was also run with only about row & column context. So it actually doesn't know all the rules yet.

The main concern with it now is that it's slow as shit because the number of masks is exponential. So we need to find a way to actually sample/trim masks in a meaningful way.

1. train a subset of masks at any time
2. reject non-useful masts
3. reject masks providing "duplicative information"