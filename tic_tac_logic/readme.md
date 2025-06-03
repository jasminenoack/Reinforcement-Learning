# Tic Tac Logic

This is a puzzle game where you need to place an X or O in every square so that every row or column has the same number of Xs and Os and there are never more than 2 of any symbol next to each other orthogonally. Additionally, you want to ensure that no two rows or columns are identical.


# RLShapingBasedAgent

This agent isn't currently working super well. It sometimes does very well and sometimes quite poorly. It's primarily dependent on really detailed shaping data to ensure that it doesn't make poor life decisions. That shaping data is incomplete. So it finds some information it's extremely confident in, but then tends to be a bit lost.

I've been testing a lot of information around shaping and scoring to try to determine what's ideal here.

Part of the problem is that a lot of the information is relatively uncertain, which leads to that information actually being misleading at least some of the time.

It's not fully clear the best way to actually recover from this
