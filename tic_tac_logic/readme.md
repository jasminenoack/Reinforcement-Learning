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

## Performance

One of the biggest issues with the mask agent is performance, he's real slow. Because of this it isn't really able to do things all that well.


         4178258270 function calls (4174764914 primitive calls) in 602.234 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     29/1    0.002    0.000  680.535  680.535 {built-in method builtins.exec}
        1    0.002    0.002  680.535  680.535 mask_tester.py:1(<module>)
     1000    0.066    0.000  680.440    0.680 mask_tester.py:76(run_episode)
   660327   27.436    0.000  674.388    0.001 mask_agent.py:69(get_applicable_masks)
    20461    0.143    0.000  670.095    0.033 mask_agent.py:222(act)
    20459    1.804    0.000  667.979    0.033 mask_agent.py:186(remove_failing_options)
   660327  167.844    0.000  570.627    0.001 masks.py:252(generate_all_patterns)
        1    0.001    0.001  300.130  300.130 mask_tester.py:102(mask_builder_view)
183114735  141.186    0.000  222.241    0.000 {method 'join' of 'str' objects}
353242080   60.893    0.000   75.264    0.000 masks.py:22(get_pattern)
124298458   21.989    0.000   60.950    0.000 masks.py:183(mask_applies)
306409304   29.662    0.000   29.662    0.000 {method 'replace' of 'str' objects}
739356943   27.707    0.000   27.707    0.000 masks.py:281(<genexpr>)
739356943   26.981    0.000   26.981    0.000 masks.py:286(<genexpr>)
739356943   26.366    0.000   26.366    0.000 masks.py:291(<genexpr>)

Ideas

1. cache the patterns so we aren't constantly recalculating them

         929679210 function calls (926990855 primitive calls) in 130.459 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     29/1    0.002    0.000  135.191  135.191 {built-in method builtins.exec}
        1    0.001    0.001  135.191  135.191 mask_tester.py:1(<module>)
     1000    0.049    0.000  135.110    0.135 mask_tester.py:76(run_episode)
    20587    0.125    0.000  131.748    0.006 mask_agent.py:222(act)
   669558   26.263    0.000  130.823    0.000 mask_agent.py:69(get_applicable_masks)
    20586    0.396    0.000  130.046    0.006 mask_agent.py:186(remove_failing_options)
        1    0.001    0.001   77.733   77.733 mask_tester.py:102(mask_builder_view)
121110499   20.256    0.000   56.046    0.000 masks.py:183(mask_applies)
   669558   10.370    0.000   33.465    0.000 masks.py:252(generate_all_patterns)
 76476886   13.636    0.000   16.947    0.000 masks.py:22(get_pattern)
 24496607    8.821    0.000   13.250    0.000 {method 'join' of 'str' objects}
 11943455    9.346    0.000    9.346    0.000 masks.py:178(remove_non_matching)
 13314359    5.216    0.000    8.135    0.000 masks.py:168(_mask_matches_pattern)
133258699    8.122    0.000    8.122    0.000 {method 'get' of 'dict' objects}
133238219    6.819    0.000    6.819    0.000 {method 'extend' of 'list' objects}
121110499    4.690    0.000    4.690    0.000 masks.py:175(_mask_matches_symbol)
 41876976    3.427    0.000    3.427    0.000 {method 'replace' of 'str' objects}
95031060/95030996    3.378    0.000    3.378    0.000 {built-in method builtins.len}
     8492    0.013    0.000    1.864    0.000 mask_agent.py:264(learn)
 11442112    1.134    0.000    1.742    0.000 {method 'add' of 'set' objects}
 40379746    1.514    0.000    1.514    0.000 masks.py:281(<genexpr>)
 40379746    1.471    0.000    1.471    0.000 masks.py:286(<genexpr>)
 40379746    1.444    0.000    1.444    0.000 masks.py:291(<genexpr>)


5. using sets instead of arrays in the mask manager

This actually made it much worse

         917205833 function calls (912813574 primitive calls) in 213.424 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     29/1    0.002    0.000  218.471  218.471 {built-in method builtins.exec}
        1    0.002    0.002  218.471  218.471 mask_tester.py:1(<module>)
     1000    0.064    0.000  218.382    0.218 mask_tester.py:76(run_episode)
   644705   58.419    0.000  214.070    0.000 mask_agent.py:70(get_applicable_masks)
    19373    0.136    0.000  213.925    0.011 mask_agent.py:221(act)
    19372    0.612    0.000  212.183    0.011 mask_agent.py:185(remove_failing_options)
        1    0.001    0.001  115.928  115.928 mask_tester.py:102(mask_builder_view)

4. Only allow useful masks to come back as "applicable"

First I added a second dict of "applicable" masks when we are pruning, this basically makes a stronger judgment about which masks to actually use when doing the calculations. I ran with just loading this to confirm it wasn't slow. It was fine.

this helped a ton

         510701248 function calls (508735218 primitive calls) in 66.603 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     29/1    0.002    0.000   71.743   71.743 {built-in method builtins.exec}
        1    0.001    0.001   71.743   71.743 mask_tester.py:1(<module>)
     1000    0.045    0.000   71.656    0.072 mask_tester.py:76(run_episode)
    19880    0.118    0.000   68.518    0.003 mask_agent.py:232(act)
   657245   15.699    0.000   68.072    0.000 mask_agent.py:70(get_applicable_masks)
    19879    0.150    0.000   66.738    0.003 mask_agent.py:196(remove_failing_options)
        1    0.011    0.011   50.553   50.553 mask_tester.py:102(mask_builder_view)

basically on learning it uses the entire set but predicting it filters to things that "feel" useful.

6. Remove duplicate masks

This has less impact than I expected

  499246687 function calls (497143723 primitive calls) in 65.431 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     30/1    0.002    0.000   70.416   70.416 {built-in method builtins.exec}
        1    0.001    0.001   70.416   70.416 mask_tester.py:1(<module>)
     1000    0.046    0.000   70.337    0.070 mask_tester.py:76(run_episode)
    20470    0.125    0.000   67.661    0.003 mask_agent.py:232(act)
   663328   15.775    0.000   66.697    0.000 mask_agent.py:70(get_applicable_masks)
    20470    0.158    0.000   65.803    0.003 mask_agent.py:196(remove_failing_options)
        1    0.001    0.001   47.738   47.738 mask_tester.py:102(mask_builder_view)
   663328   11.042    0.000   35.725    0.000 masks.py:268(generate_all_patterns)
 12157895    8.364    0.000   13.080    0.000 {method 'join' of 'str' objects}
130891945    7.224    0.000    7.224    0.000 {method 'get' of 'dict' objects}
130872370    6.007    0.000    6.007    0.000 {method 'extend' of 'list' objects}
 18256576    3.155    0.000    3.904    0.000 masks.py:22(get_pattern)
 12428228    1.245    0.000    1.961    0.000 {method 'add' of 'set' objects}
  2300649    0.559    0.000    1.959    0.000 masks.py:183(mask_applies)
 17004689    1.739    0.000    1.739    0.000 {method 'replace' of 'str' objects}
 43211430    1.611    0.000    1.611    0.000 masks.py:297(<genexpr>)
 43211430    1.570    0.000    1.570    0.000 masks.py:302(<genexpr>)
 43211430    1.534    0.000    1.534    0.000 masks.py:307(<genexpr>)
    20470    0.397    0.000    1.373    0.000 mask_agent.py:175(find_aggressive_failures)



2. cache applicable masks to avoid rechoosing every time


3. short circuit best option, if you have one that's plenty


